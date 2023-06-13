from network_Andersen import cnn_lstm
from confidence_interval_demo import *
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.signal as signal
import argparse
import sklearn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 根据rri_data与label计算房颤片段长度
def label2length(data_, label_):
    data = deepcopy(data_)
    label = deepcopy(label_)
    label_padded = np.zeros(len(label)+2)
    label_padded[1:-1] = label[:]
    label_diff = np.diff(label_padded)
    index = np.linspace(0,len(label),len(label)+1).astype('int')
    starts = index[label_diff==1]
    ends = index[label_diff==-1]
    assert len(starts) == len(ends)
    for s,e in zip(starts, ends):
        length = np.sum(data[s:e])
        label[s:e] *= length
    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', type=str) # AFDB数据 .csv文件的路径
    parser.add_argument('-model_save_name', type=str) # 五折交叉验证训练的模型保存路径
    parser.add_argument('-labels_save_path', type=str) # 预测结果以及标签的保存路径
    parser.add_argument('-result_save_path', type=str) # 统计结果保存路径
    parser.add_argument('-filter', type=bool) # 如果 True 对结果进行中值滤波

    args = parser.parse_args()

    data_path = args.data_path
    model_save_name = args.model_save_name
    labels_save_path = args.labels_save_path
    result_save_path = args.result_save_path
    filter = args.filter
    # data_path = '/nas/liuyuhang/ecg_af_rri/data/afdb_csv'
    # model_save_name = './models/afdb/cnn_lstm'
    # labels_save_path = '/nas/liuyuhang/ecg_af_rri/save/Andersen/AFDB'
    # result_save_path = './results/Andesen/AFDB'

    # Train
    results = []
    label_list = []
    pred_list = []

    len_rri = 30
    afdb_5fold_list = pd.read_csv('./afdb_5fold_list.csv', header=[0], index_col=None, dtype='str')
    for fold in range(5):
        record_test = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values=='%s'%(fold+1)]
        model = cnn_lstm()
        model.load_weights('%s_fold%d.h5' % (model_save_name, fold+1))
        for record in record_test:
            holter = pd.read_csv('%s/%s.csv' % (data_path, record))
            holter_data = holter['rri'].values / 1000
            holter_label = holter['label'].values
            holter_label[holter_label >= 1] = 1
            holter_label[holter_label <= 0] = 0
            len_holter = len(holter_data)
            holter_data_padded = np.zeros(int(np.ceil(len_holter/len_rri)*len_rri))
            holter_data_padded[0:len_holter] = holter_data
            holter_data_padded[len_holter:] = holter_data[-1]
            segmented_data = np.reshape(holter_data_padded, (-1,len_rri,1))
            segmented_pred = model.predict(segmented_data)
            if filter:
                segmented_pred = signal.medfilt(segmented_pred, kernel_size=(3,1))
            holter_pred = np.repeat(segmented_pred, len_rri, axis=1).flatten()[0:len_holter]
            np.save('%s/%s_label.npy' % (labels_save_path, record), holter_label)
            if filter:
                np.save('%s/%s_pred_filtered.npy' % (labels_save_path, record), holter_pred)
            else:
                np.save('%s/%s_pred.npy' % (labels_save_path, record), holter_pred)
            

            label_list.append(holter_label)
            pred_list.append(holter_pred)
            [sensitivity, specificity, accuracy] = data_acc(holter_label, holter_pred)

            if np.sum(holter_label==0) and np.sum(holter_label==1):
                [AUC] = data_auc(holter_label, holter_pred)
            else:
                AUC = -1
            result = [record, sensitivity, specificity, accuracy, -1, -1, AUC, -1, -1]
            results.append(result)

    pred_total = np.concatenate(pred_list)
    label_total = np.concatenate(label_list)

    # 所有心拍的统计结果
    [sensitivity, specificity, accuracy] = data_acc(label_total, pred_total)
    [acc_lower, acc_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_acc)
    [AUC] = data_auc(label_total, pred_total)
    [AUC_lower, AUC_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_auc)
    result = ['total', sensitivity, specificity, accuracy, acc_lower, acc_higher, AUC, AUC_lower, AUC_higher]
    results.append(result)
    pd.DataFrame(results).to_csv('%s/afdb_results.csv' % (result_save_path), header=['record', 'sensitivity', 'specificity', 'accuracy', 'acc_lower', 'acc_higher', 'AUC', 'AUC_lower', 'AUC_higher'], index=None, mode='w')
    fp, tp, th = sklearn.metrics.roc_curve(label_total, pred_total)
    pd.DataFrame([fp, tp, th]).T.to_csv('%s/afdb_roc.csv' % (result_save_path), header=['fpr', 'tpr', 'thresholds'], index=None, mode='w')


    # 小于30s房颤片段的心拍的统计结果
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_below30s = []
    preds_below30s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
        if filter:
            pred = np.load('%s/%s_pred_filtered.npy' % (labels_save_path, record))
        else:
            pred = np.load('%s/%s_pred.npy' % (labels_save_path, record))
        cur_l = label[np.logical_and(label_af_length>0.1, label_af_length<=30)]
        cur_p = pred[np.logical_and(label_af_length>0.1, label_af_length<=30)]
        if len(cur_l)==0:
            continue
        labels_below30s.append(cur_l)
        preds_below30s.append(cur_p)

        [sensitivity, specificity, accuracy] = data_acc(cur_l, cur_p)
        result = [record, sensitivity, specificity, accuracy, -1, -1]
        results.append(result)

    label_total = np.concatenate(labels_below30s)
    pred_total = np.concatenate(preds_below30s)
    [sensitivity, specificity, accuracy] = data_acc(label_total, pred_total)
    [acc_lower, acc_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_acc)
    result = ['total', sensitivity, specificity, accuracy, acc_lower, acc_higher]
    results.append(result)
    pd.DataFrame(results).to_csv('%s/afdb_below30s_results.csv' % (result_save_path), header=['record', 'sensitivity', 'specificity', 'accuracy', 'acc_lower', 'acc_higher'], index=None, mode='w')

    # 30s-60s房颤片段的心拍的统计结果
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_30to60s = []
    preds_30to60s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
        if filter:
            pred = np.load('%s/%s_pred_filtered.npy' % (labels_save_path, record))
        else:
            pred = np.load('%s/%s_pred.npy' % (labels_save_path, record))
        cur_l = label[np.logical_and(label_af_length>30, label_af_length<=60)]
        cur_p = pred[np.logical_and(label_af_length>30, label_af_length<=60)]
        if len(cur_l)==0:
            continue
        labels_30to60s.append(cur_l)
        preds_30to60s.append(cur_p)

        [sensitivity, specificity, accuracy] = data_acc(cur_l, cur_p)
        result = [record, sensitivity, specificity, accuracy, -1, -1]
        results.append(result)

    label_total = np.concatenate(labels_30to60s)
    pred_total = np.concatenate(preds_30to60s)
    [sensitivity, specificity, accuracy] = data_acc(label_total, pred_total)
    [acc_lower, acc_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_acc)
    result = ['total', sensitivity, specificity, accuracy, acc_lower, acc_higher]
    results.append(result)
    pd.DataFrame(results).to_csv('%s/afdb_30to60s_results.csv' % (result_save_path), header=['record', 'sensitivity', 'specificity', 'accuracy', 'acc_lower', 'acc_higher'], index=None, mode='w')

    # 大于60s房颤片段的心拍的统计结果
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_over60s = []
    preds_over60s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
        if filter:
            pred = np.load('%s/%s_pred_filtered.npy' % (labels_save_path, record))
        else:
            pred = np.load('%s/%s_pred.npy' % (labels_save_path, record))
        cur_l = label[label_af_length>60]
        cur_p = pred[label_af_length>60]
        if len(cur_l)==0:
            continue
        labels_over60s.append(cur_l)
        preds_over60s.append(cur_p)

        [sensitivity, specificity, accuracy] = data_acc(cur_l, cur_p)
        result = [record, sensitivity, specificity, accuracy, -1, -1]
        results.append(result)

    label_total = np.concatenate(labels_over60s)
    pred_total = np.concatenate(preds_over60s)
    [sensitivity, specificity, accuracy] = data_acc(label_total, pred_total)
    [acc_lower, acc_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_acc)
    result = ['total', sensitivity, specificity, accuracy, acc_lower, acc_higher]
    results.append(result)
    pd.DataFrame(results).to_csv('%s/afdb_over60s_results.csv' % (result_save_path), header=['record', 'sensitivity', 'specificity', 'accuracy', 'acc_lower', 'acc_higher'], index=None, mode='w')

