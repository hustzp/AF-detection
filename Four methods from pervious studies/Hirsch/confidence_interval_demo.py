# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap(label, pred, B, c, func):
    """
    obtain bootstrap confidence interval
    :param B: sampling frequency B>=1000
    :param c: confidence level
    :param func: Sample Estimator
    :return: Upper and lower limits of confidence intervals
    """
    n = len(label)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)

        label_sample = label[index_arr]
        pred_sample = pred[index_arr]

        sample_result = func(label_sample, pred_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    sample_result_arr = np.asarray(sample_result_arr)
    for j in range(sample_result_arr.shape[1]):
        cur_sample_result_arr = sample_result_arr[:, j]
        auc_sample_arr_sorted = sorted(cur_sample_result_arr)
        lower = auc_sample_arr_sorted[k1]
        higher = auc_sample_arr_sorted[k2]
        print(lower, higher)
    return lower, higher


def data_acc(label, pred):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    p = pred[label >= 1]
    tp = p[p == 1]
    n = pred[label <= 0]
    tn = n[n == 0]

    return [(tp.shape[0] + tn.shape[0]) / (p.shape[0] + n.shape[0])]


def data_auc(label, pred):
    try:
        auc = roc_auc_score(y_true=label, y_score=pred)
    except ValueError:
        auc = 0
    return [auc]


if __name__ == '__main__':
    label = np.array([1, 0])
    pred = np.array([0.8, 0.2])

    print('AUC')
    print(data_auc(label, pred))
    bootstrap(label, pred, 1000, 0.95, data_auc)

    print('ACC')
    print(data_acc(label, pred))
    bootstrap(label, pred, 1000, 0.95, data_acc)
