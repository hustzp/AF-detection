import numpy as np
import os
import argparse
import pandas as pd
import feature_tools as ft


class FileExtract:
    def __init__(self):
        self.data = None
        self.file = None
        self.time = None
        self.start_time = None
        self.label = None

    def extract(self, file):
        self.file = file
        file_path = file

        assert os.path.exists(file_path)

        self.label = []

        with open(file_path, encoding='gbk') as f:
            self.data = f.readlines()

        self.start_time = self._get_start_time()
        self._get_label()

        self._data_process()

        return self.seq, self.label_seq, self.time

    def _get_start_time(self):
        for index, line in enumerate(self.data):
            if line.startswith('Start time'):
                line = line.split(':')
                return int(line[1]) * 3600 + int(line[2]) * 60

    def _get_label(self):
        i = 4
        while self.data[i][:2] != 'RR' and self.data[i] != '\n':
            cur = self.data[i].strip()
            if cur[-1] == '0' or cur[-1] == ')':
                time1 = int(cur[:2]) * 3600 + int(cur[3:5]) * 60 + int(cur[6:8])
                second = int(cur.split('(')[-1].split(')')[0])
                if time1 < self.start_time:
                    time1 += 24 * 3600
                self.label.append([time1, time1 + second])
            i += 1

    def _data_process(self):
        i = 0
        for line in self.data:
            i += 1
            if line.strip() == 'End header':
                break

        self.time = np.asarray([int(line[1:3]) * 3600 + int(line[4:6]) * 60 + int(line[7:9]) for line in self.data[i:]])
        self.time[self.time < self.start_time] += 24 * 3600
        self.seq = np.asarray([line.strip().split(' ')[0].split(']')[-1][1:]
                               if len(line.strip().split(' ')[0].split(']')[-1]) > 0 else 0 for line in
                               self.data[i:]], dtype=int)
        total_sign = np.asarray([line.strip().split(' ')[0].split(']')[-1][0]
                                 if len(line.strip().split(' ')[0].split(']')[-1]) > 0 else 'Z' for line in
                                 self.data[i:]])

        assert self.time.shape[0] == self.seq.shape[0]

        self.label_seq = np.zeros_like(self.seq)

        for each in self.label:
            self.label_seq[(self.time >= each[0]) & (self.time <= each[1])] = 1

        self.label_seq[total_sign == 'Z'] = -1
        self.label_seq[(self.seq < 200) | (self.seq > 2000)] = -1


class FileExtract2:
    def __init__(self):
        self.data = None
        self.seq = None
        self.label = None
        self.t_list = None

        self.file = None

    def extract(self, file):
        self.file = file
        file_path = file

        assert os.path.exists(file_path)

        with open(file_path, encoding='gbk') as f:
            self.data = f.readlines()

        self._data_process()

        return self.seq, self.label

    def _data_process(self):
        self.seq = np.asarray([line.strip().split(' ')[0][1:] for line in self.data], dtype=np.int64)
        self.t_list = np.asarray([line.strip().split(' ')[0][0] for line in self.data])

        # self.seq[self.t_list == 'Z'] = 0
        # self.seq[self.t_list == 'A'] = 0
        # self.seq[self.t_list == 'V'] = 0

        self.label = np.zeros(shape=(self.seq.shape[0],))
        self.label[(self.seq > 2000) | (self.seq < 200)] = -1
        self.label[self.t_list == 'Z'] = -1


def chunk_file_read_and_feature_extraction(file):
    path_num = len(file)
    train_data = []
    # file=path
    file_extract_1 = FileExtract()
    file_extract_2 = FileExtract2()
    for i in range(path_num):
        if '_' in file:
            seq, y = file_extract_1.extract(file)
            seq_feature, seq_label = ft.feature_extraction(RRI=seq, label=y)
        else:
            seq, y = file_extract_2.extract(file)
            seq_feature, seq_label = ft.feature_extraction(RRI=seq, label=y)

        if train_data == []:
            train_data = seq_feature
            train_label = seq_label
        else:
            train_data = np.concatenate([train_data, seq_feature], axis=0)
            train_label = np.concatenate([train_label, seq_label], axis=0)
    return train_data, train_label


def chunk_read_and_feature_extraction(data_path, file):
    path_num = len(file)
    train_data = []

    for i in range(path_num):
        RRI = pd.read_csv(data_path + 'afdb/' + file).to_numpy()[:, 1]
        label = pd.read_csv(data_path + 'afdb/' + file).to_numpy()[:, 2]
        seq_feature, seq_label = ft.feature_extraction(RRI=RRI, label=label)

        if train_data:
            train_data = seq_feature
            train_label = seq_label
        else:
            train_data = np.concatenate([train_data, seq_feature], axis=0)
            train_label = np.concatenate([train_label, seq_label], axis=0)
    return train_data, train_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', type=str)
    parser.add_argument('-save_path', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path

    file_extract_1 = FileExtract()
    file_extract_2 = FileExtract2()

    files = os.listdir(data_path)

    interval = 90

    for index, file in enumerate(files):
        save_name = file.split('.')[0].split('_')[0]
        print(index, save_name)

        if os.path.exists(save_path + '/' + '{}_train_x.npy'.format(save_name)) and \
                os.path.exists(save_path + '/' + '{}_train_y.npy'.format(save_name)):
            continue

        if '_' in file:
            seq, y = file_extract_1.extract(data_path, file)
        else:
            seq, y = file_extract_2.extract(data_path, file)

        c_x = np.reshape(seq[:seq.shape[0] // interval * interval], (-1, interval))
        c_y = np.reshape(y[:y.shape[0] // interval * interval], (-1, interval))

        c_x = c_x[(c_y != -1).all(axis=-1)]
        c_y = c_y[(c_y != -1).all(axis=-1)]

        print(c_x.shape, c_y.shape)

        np.save(save_path + '/' + '{}_train_x.npy'.format(save_name), c_x)
        np.save(save_path + '/' + '{}_train_y.npy'.format(save_name), c_y)
