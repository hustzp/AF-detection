import numpy as np
from confidence_interval_demo import bootstrap, data_acc, data_auc


def metric(y_true, y_pred):
    l = y_true.shape[0]
    TP = 0
    TN = 0
    FP = 1e-5
    FN = 1e-5
    for i in range(l):
        if (y_true[i] == 1) & (y_pred[i] == 1):
            TP += 1
        if (y_true[i] == 0) & (y_pred[i] == 1):
            FP += 1
        if (y_true[i] == 1) & (y_pred[i] == 0):
            FN += 1
        if (y_true[i] == 0) & (y_pred[i] == 0):
            TN += 1
    acc = (TP + TN) / (TP + TN + FP + FN)
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    return acc, Sn, Sp, F1


def piece_detect_rate(predict_label, predicted_prob, raw_label, time):
    l = raw_label.shape[0]
    s = predict_label.shape[0]
    recovered_pred_label = np.zeros(raw_label.shape)
    recovered_pred_prob = np.zeros(raw_label.shape)
    AF_0_30 = 0
    AF_30_60 = 0
    AF_60 = 0
    AF_length_30_sum = 0
    AF_length_30_60_sum = 0
    AF_length_60_sum = 0
    AF_dect_num_30 = 0
    AF_dect_num_30_60 = 0
    AF_dect_num_60 = 0
    AF_pieces_30 = []
    AF_pieces_30_60 = []
    AF_pieces_60 = []

    continues = 0
    for i in range(s):
        recovered_pred_prob[15 * i:15 * i + 30] += predicted_prob[i, 1] * np.ones(30)
        if predict_label[i] == 1:
            recovered_pred_label[15 * i:15 * i + 30] = 1
        else:
            recovered_pred_label[15 * i:15 * i + 30] = 0
    recovered_pred_prob[15:(raw_label.shape[0] - 15)] /= 2
    for i in range(1, l - 1):
        if 1 < i < l - 1:
            if (raw_label[i] == 1) and (raw_label[i - 1] == 0):
                t_s = time[i]
                i_s = i
                continues = 1
            if (continues == 1) and (raw_label[i] == 1) and (raw_label[i + 1] == 0):
                t_e = time[i + 1]
                i_e = i + 1
                continues = 0
                if t_e > t_s:
                    interval = np.asarray([t_s, t_e - t_s])
                    if t_e - t_s < 30:
                        AF_dect_num_30 += np.sum(recovered_pred_label[i_s:i_e] == raw_label[i_s:i_e])
                        AF_length_30_sum += i_e - i_s
                        AF_0_30 += 1
                        piece = np.stack((recovered_pred_prob[i_s:i_e], raw_label[i_s:i_e]), axis=0)
                        if AF_pieces_30 == []:
                            AF_pieces_30 = piece
                        else:
                            AF_pieces_30 = np.concatenate([AF_pieces_30, piece], axis=1)
                    if 30 < t_e - t_s < 60:
                        AF_dect_num_30_60 += np.sum(recovered_pred_label[i_s:i_e] == raw_label[i_s:i_e])
                        AF_length_30_60_sum += i_e - i_s
                        AF_30_60 += 1
                        piece = np.stack((recovered_pred_prob[i_s:i_e], raw_label[i_s:i_e]), axis=0)
                        if AF_pieces_30_60 == []:
                            AF_pieces_30_60 = piece
                        else:
                            AF_pieces_30_60 = np.concatenate([AF_pieces_30_60, piece], axis=1)
                    if t_e - t_s > 60:
                        AF_dect_num_60 += np.sum(recovered_pred_label[i_s:i_e] == raw_label[i_s:i_e])
                        AF_length_60_sum += i_e - i_s
                        AF_60 += 1
                        piece = np.stack((recovered_pred_prob[i_s:i_e], raw_label[i_s:i_e]), axis=0)
                        if AF_pieces_60 == []:
                            AF_pieces_60 = piece
                        else:
                            AF_pieces_60 = np.concatenate([AF_pieces_60, piece], axis=1)
    AF_nums = [AF_0_30, AF_30_60, AF_60]
    if AF_length_30_sum == 0:
        rate_30 = -1
    else:
        rate_30 = AF_dect_num_30 / AF_length_30_sum
    if AF_length_30_60_sum == 0:
        rate_30_60 = -1
    else:
        rate_30_60 = AF_dect_num_30_60 / AF_length_30_60_sum
    if AF_length_60_sum == 0:
        rate_60 = -1
    else:
        rate_60 = AF_dect_num_60 / AF_length_60_sum

    # acc& bootstrap
    if AF_pieces_30 == []:
        acc_30_lower = -1
        acc_30_higher = -1
    else:
        AF_pieces_30_label = 1 * (AF_pieces_30[0, :] >= 0.5)
        acc_30_lower, acc_30_higher = bootstrap(label=AF_pieces_30[1, :], pred=AF_pieces_30_label, B=1000, c=0.95,
                                                func=data_acc)

    if AF_pieces_30_60 == []:
        acc_30_60_lower = -1
        acc_30_60_higher = -1
    else:
        AF_pieces_30_60_label = 1 * (AF_pieces_30_60[0, :] >= 0.5)
        acc_30_60_lower, acc_30_60_higher = bootstrap(label=AF_pieces_30_60[1, :], pred=AF_pieces_30_60_label, B=1000,
                                                      c=0.95, func=data_acc)

    if AF_pieces_60 == []:
        acc_60_lower = -1
        acc_60_higher = -1
    else:
        AF_pieces_60_label = 1 * (AF_pieces_60[0, :] >= 0.5)
        acc_60_lower, acc_60_higher = bootstrap(label=AF_pieces_60[1, :], pred=AF_pieces_60_label, B=1000, c=0.95,
                                                func=data_acc)

    return AF_nums, [rate_30, acc_30_lower, acc_30_higher], \
           [rate_30_60, acc_30_60_lower, acc_30_60_higher], \
           [rate_60, acc_60_lower, acc_60_higher], \
           AF_pieces_30, AF_pieces_30_60, AF_pieces_60
