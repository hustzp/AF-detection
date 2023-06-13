from method_Islam import HBA
from confidence_interval_demo import *
import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', type=str) # AFDB数据 .csv文件的路径
    parser.add_argument('-save_name', type=str) # 阈值搜寻结果保存路径

    args = parser.parse_args()

    data_path = args.data_path
    save_name = args.save_name
    # data_path = '/nas/liuyuhang/ecg_af_rri/data/afdb_csv'
    # save_name = './results/Islam/AFDB/threshold_searching'

    # Train
    afdb_5fold_list = pd.read_csv('./afdb_5fold_list.csv', header=[0], index_col=None, dtype='str')
    for fold in range(5):
        print('Fold%d Threshold Searching' % (fold+1))
        record_train = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values!='%s'%(fold+1)]
        record_test = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values=='%s'%(fold+1)]

        data_list = []
        label_list = []
        for record in record_train:
            holter = pd.read_csv('%s/%s.csv' % (data_path, record))
            c_x= holter['rri'].values
            c_y = holter['label'].values
            data_list.append(c_x)
            label_list.append(c_y)
        label = np.concatenate(label_list)

        Metrics = []
        Thresholds = np.linspace(1,3.5,26)
        for n, threshold in enumerate(Thresholds):
            pred_list = []
            # search best threshold
            for c_x in data_list:
                c_p = HBA(c_x, threshold, window_size=70, step_size=70)
                pred_list.append(c_p)
            pred = np.concatenate(pred_list)
            [_, _, acc] = data_acc(label, pred)
            Metrics.append(acc)
            print('%-2d threshold %.1f acc %.2f' % (n, threshold, acc*100))
        
        best_threshold = Thresholds[np.argmax(Metrics)]
        best_acc = Metrics[np.argmax(Metrics)]
        print('Best threshold: %.1f & Best acc: %.2f' % (best_threshold, best_acc*100))
        pd.DataFrame([Thresholds, Metrics]).T.to_csv('%s_fold%d.csv' % (save_name, fold+1), header=['Threshold', 'Acc'], index=None, mode='w')
