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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', type=str) # data of AFDB, pathway of the file of .csv
    parser.add_argument('-segment_data_save_path', type=str) # The data pathway after sliding sampling of AFDB dataset
    parser.add_argument('-save_name', type=str) # pathway of the model

    args = parser.parse_args()

    data_path = args.data_path
    segment_data_save_path = args.segment_data_save_path
    save_name = args.save_name
    # data_path = '/nas/liuyuhang/ecg_af_rri/data/afdb_csv'
    # segment_data_save_path = '/nas/liuyuhang/ecg_af_rri/data/afdb_npy'
    # save_name = './models/afdb/cnn_lstm'

    # recording were divided into 30 RRIs sliding window with step of 10 RRIs
    len_rri = 30
    sample_shift = 10
    af_threshold = 0.5
    records = os.listdir('./data/afdb_csv/')
    for record in records:
        data_holter = pd.read_csv('./data/afdb_csv/%s' % (record))
        len_holter = len(data_holter['rri'].values)
        num_segment = np.ceil((len_holter-len_rri)/sample_shift + 1).astype('int')
        data_rri = np.zeros((num_segment,len_rri,1))
        label_rri = np.zeros((num_segment,1))
        for i in range(num_segment):
            start = i*sample_shift
            start = len_holter-len_rri if start>(len_holter-len_rri) else start
            end = start + 30
            data_rri[i,:,0] = data_holter['rri'].values[start:end] / 1000 # covert ms to s
            label_rri[i,0] = 1 if np.sum(data_holter['label'].values[start:end])>=af_threshold*len_rri else 0

        np.save('%s/%s_%drri_data.npy' % (segment_data_save_path, record[0:-4], len_rri), data_rri)
        np.save('%s/%s_%drri_label.npy' % (segment_data_save_path, record[0:-4], len_rri), label_rri)

    # Train
    afdb_5fold_list = pd.read_csv('./afdb_5fold_list.csv', header=[0], index_col=None, dtype='str')
    for fold in range(5):
        record_train = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values!='%s'%(fold+1)]
        record_test = afdb_5fold_list['record'].values[afdb_5fold_list['fold'].values=='%s'%(fold+1)]
        x_train = np.concatenate([np.load('%s/%s_%drri_data.npy' % (segment_data_save_path, record, len_rri)) for record in record_train])
        y_train = np.concatenate([np.load('%s/%s_%drri_label.npy' % (segment_data_save_path, record, len_rri)) for record in record_train])
        print('Training CNN-LSTM Model at Fold %d' % (fold+1))
        model = cnn_lstm()
        model.fit(x_train, y_train , batch_size=256, epochs=50, shuffle=True)
        model.save('%s_fold%d.h5' % (save_name, fold+1))
