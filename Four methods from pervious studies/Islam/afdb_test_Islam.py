from method_Islam import HBA
from confidence_interval_demo import *
from copy import deepcopy
import numpy as np
import pandas as pd
import argparse
import sklearn


# obtain the length of AF episodes according to the rri_data and labels
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

    parser.add_argument('-data_path', type=str) # AFDB data, path of the files .csv
    parser.add_argument('-ts_save_name', type=str) # path of threshold results
    parser.add_argument('-labels_save_path', type=str) # path of predicted results and labels
    parser.add_argument('-result_save_path', type=str)  # path of results

    args = parser.parse_args()

    data_path = args.data_path
    ts_save_name = args.ts_save_name
    labels_save_path = args.labels_save_path
    result_save_path = args.result_save_path
    # data_path = '/nas/liuyuhang/ecg_af_rri/data/afdb_csv'
    # ts_save_name = './results/Islam/AFDB/threshold_searching'
    # labels_save_path = '/nas/liuyuhang/ecg_af_rri/save/Islam/AFDB'
    # result_save_path = './results/Islam/AFDB'

    # Train
    results = []
    label_list = []
    pred_list = []

    afdb_5fold_list = pd.read_csv('./afdb_5fold_list.csv', header=[0], index_col=None, dtype='str')
    for fold in range(5):
        # obtain the best threshold in testing sets
        ts_result = pd.read_csv('%s_fold%d.csv' % (ts_save_name, fold+1))
        Thresholds = ts_result['Threshold'].values
        Metrics = ts_result['Acc'].values
        best_threshold = Thresholds[np.argmax(Metrics)]
        print('Fold%d Testing with Best threshold: %.1f' % (fold+1, best_threshold))
        record_train = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values!='%s'%(fold+1)]
        record_test = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values=='%s'%(fold+1)]

        for record in record_train:
            holter = pd.read_csv('%s/%s.csv' % (data_path, record))
            c_x = holter['rri'].values
            c_y = holter['label'].values
            c_p = HBA(c_x, best_threshold, window_size=70, step_size=70)
            np.save('%s/%s_label.npy' % (labels_save_path, record), c_y)
            np.save('%s/%s_pred.npy' % (labels_save_path, record), c_p)
            af_length = label2length(c_x/1000, c_y)
            np.save('%s/%s_label_af_length.npy' % (labels_save_path, record), af_length)

            label_list.append(c_y)
            pred_list.append(c_p)
            [sensitivity, specificity, accuracy] = data_acc(c_y, c_p)

            if np.sum(c_y==0) and np.sum(c_y==1):
                [AUC] = data_auc(c_y, c_p)
            else:
                AUC = -1
            result = [record, sensitivity, specificity, accuracy, -1, -1, AUC, -1, -1]
            results.append(result)

    pred_total = np.concatenate(pred_list)
    label_total = np.concatenate(label_list)

    # results on heartbeats
    [sensitivity, specificity, accuracy] = data_acc(label_total, pred_total)
    [acc_lower, acc_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_acc)
    [AUC] = data_auc(label_total, pred_total)
    [AUC_lower, AUC_higher] = bootstrap(label_total, pred_total, 1000, 0.95, data_auc)
    result = ['total', sensitivity, specificity, accuracy, acc_lower, acc_higher, AUC, AUC_lower, AUC_higher]
    results.append(result)
    pd.DataFrame(results).to_csv('%s/afdb_results.csv' % (result_save_path), header=['record', 'sensitivity', 'specificity', 'accuracy', 'acc_lower', 'acc_higher', 'AUC', 'AUC_lower', 'AUC_higher'], index=None, mode='w')

    fp, tp, th = sklearn.metrics.roc_curve(label_total, pred_total)
    pd.DataFrame([fp, tp, th]).T.to_csv('%s/afdb_roc.csv' % (result_save_path), header=['fpr', 'tpr', 'thresholds'], index=None, mode='w')


    # results of AF episodes of less than 30 s
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_below30s = []
    preds_below30s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
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

    # results of AF episodes of 30s-60s
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_30to60s = []
    preds_30to60s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
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

    # results of AF episodes of more than 60s
    records = list(set([file.split('_')[0] for file in os.listdir(labels_save_path)]))
    results = []
    labels_over60s = []
    preds_over60s = []
    for record in records:
        label_af_length = np.load('%s/%s_label_af_length.npy' % (labels_save_path, record))
        label = np.load('%s/%s_label.npy' % (labels_save_path, record))
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

