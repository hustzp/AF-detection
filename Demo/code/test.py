
import numpy as np
import os
from Demo.code.model import build_model_lstm
from Demo.code.load_data import load_data_afdb, load_data_mit, load_data_nsr, load_data_nsrrri


def test(data_path, model_path, length=90):
    match = ['AFDB', 'MITDB', 'NSRDB', 'NSRRRIDB']
    model = build_model_lstm((length, 1))
    model.load_weights(model_path)

    save_beat = [0, 0, 0, 0]
    save_patient = [np.array([]), np.array([])]

    for m in match:
        cur_path = data_path + '/' + m

        with open(cur_path + '/' + 'RECORDS') as f:
            filenames = f.readlines()
        filenames = [file.strip() for file in filenames]

        cur_save_beat = [0, 0, 0, 0]
        cur_save_patient = [np.array([]), np.array([])]

        for file in filenames:
            print('{} {} running'.format(m, file))

            if m == 'AFDB':
                seq, label = load_data_afdb(cur_path, file)
            elif m == 'MITDB':
                seq, label = load_data_mit(cur_path, file)
            elif m == 'NSRDB':
                seq, label = load_data_nsr(cur_path, file)
            else:
                seq, label = load_data_nsrrri(cur_path, file)

            pred = test_seq(seq, label, length, model)

            re_beat = beat_level_metric(label, pred)
            re_patient = patient_level_metric(seq, pred, label)

            cur_save_beat[0] += re_beat[0]
            cur_save_beat[1] += re_beat[1]
            cur_save_beat[2] += re_beat[2]
            cur_save_beat[3] += re_beat[3]

            if (label >= 1).any():
                cur_save_patient[0] = np.hstack((cur_save_patient[0], re_patient))
            else:
                cur_save_patient[1] = np.hstack((cur_save_patient[1], re_patient))

        save_beat[0] += cur_save_beat[0]
        save_beat[1] += cur_save_beat[1]
        save_beat[2] += cur_save_beat[2]
        save_beat[3] += cur_save_beat[3]

        save_patient[0] = np.hstack((save_patient[0], cur_save_patient[0]))
        save_patient[1] = np.hstack((save_patient[1], cur_save_patient[1]))

    return save_beat, save_patient


def test_seq(sample, label, length, model, cur_inter=10, end=3):
    pred_save = []
    for i in range(0, length, cur_inter):
        if len(pred_save) == end:
            break

        cur_sample = sample[i:]
        if cur_sample.shape[0] < length:
            break
        pre_pred = np.zeros(shape=(i,))

        cur_t_x = np.reshape(cur_sample[:cur_sample.shape[0] // length * length], (-1, length))
        cur_s_x = cur_sample[-length:]
        cur_interval = cur_sample.shape[0] - cur_sample.shape[0] // length * length

        cur_pred = np.hstack((pre_pred, model_test(np.vstack((cur_t_x, cur_s_x)), model, cur_interval)))
        assert cur_pred.shape[0] == label.shape[0]

        pred_save.append(cur_pred)

    pred_save = np.asarray(pred_save)
    total_pred = np.zeros(shape=(pred_save.shape[1], ))

    for i in range(pred_save.shape[1]):
        com = pred_save[:i//cur_inter+1, i]
        if com[com >= 0.5].shape[0] >= com.shape[0] / 2:
            total_pred[i] = np.max(com)
        else:
            total_pred[i] = np.min(com)

    return total_pred


def model_test(test_x, model, interval):
    predict_y = model.predict(np.expand_dims(test_x, 2))
    predict_y = np.squeeze(predict_y)

    s_y = predict_y[-1]
    predict_y = predict_y[:-1].ravel()
    if interval > 0:
        predict_y = np.hstack((predict_y, s_y[-interval:]))

    return predict_y


def beat_level_metric(label, pred):
    p = pred[label >= 1]
    tp = p[p >= 0.5]
    n = pred[label == 0]
    tn = n[n < 0.5]

    return [tp.shape[0], p.shape[0], tn.shape[0], n.shape[0]]


def patient_level_metric(seq, pred, label):
    i = 0

    while i < seq.shape[0]:
        if label[i] > -0.5 and pred[i] >= 0.5:
            e = i + 1
            while e < seq.shape[0] and label[e] > -0.5 and pred[e] >= 0.5:
                e += 1

            if np.sum(seq[i:e]) >= 30:
                return 1

            i = e
        i += 1

    return 0
