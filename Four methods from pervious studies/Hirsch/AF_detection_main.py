import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
import feature_tools as ft
import metrics
from metrics import metric
import warnings

warnings.filterwarnings("ignore")

path = '/root/AF_detection/'
train_file_num = 16
test_file_num = 7
folds = 4
acc = np.zeros(test_file_num)
Sp = np.zeros(test_file_num)
Sn = np.zeros(test_file_num)
F1 = np.zeros(test_file_num)
total_label = []

for j in range(folds):
    print('fold' + str(j) + ' begins')
    train_data = []
    test_data = []
    filenames = ft.fold_filenames(fold_index=j + 1)
    train_files = filenames[:train_file_num]
    test_files = filenames[train_file_num:train_file_num + test_file_num]
    for i in range(train_file_num):
        print('train file' + str(i) + ' read')
        if train_data == []:
            RRI = pd.read_csv(path + 'afdb/ ' + train_files[i]).to_numpy()[:, 1]
            label = pd.read_csv(path + 'afdb/' + train_files[i]).to_numpy()[:, 2]
            train_data, train_label = ft.feature_extraction(RRI=RRI, label=label)
        else:
            RRI = pd.read_csv(path + 'afdb/' + train_files[i]).to_numpy()[:, 1]
            label = pd.read_csv(path + 'afdb/' + train_files[i]).to_numpy()[:, 2]
            data, label = ft.feature_extraction(RRI=RRI, label=label)
            train_data = np.concatenate([train_data, data], axis=0)
            train_label = np.concatenate([train_label, label], axis=0)

    print("training")
    model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=80)
    model.fit(train_data, train_label)
    print(model)

    acc = np.zeros(test_file_num)
    Sp = np.zeros(test_file_num)
    Sn = np.zeros(test_file_num)
    F1 = np.zeros(test_file_num)
    AUC = np.zeros(test_file_num)
    AF_nums = np.zeros((test_file_num, 3))
    AF_dect_rate = np.zeros((test_file_num, 3))
    for i in range(len(test_files)):
        print('test file' + str(i) + ' read')
        test_data = []
        if test_data == []:
            RRI = pd.read_csv(path + 'afdb/' + test_files[i]).to_numpy()[:, 1]
            label = pd.read_csv(path + 'afdb/' + test_files[i]).to_numpy()[:, 2]
            time = pd.read_csv(path + 'afdb/' + test_files[i]).to_numpy()[:, 0]
            test_data, test_label = ft.feature_extraction(RRI=RRI, label=label)
        else:
            RRI = pd.read_csv(path + test_files[i]).to_numpy()[:, 1]
            label = pd.read_csv(path + test_files[i]).to_numpy()[:, 2]
            data, label = ft.feature_extraction(RRI=RRI, label=label)
            test_data = np.concatenate([test_data, data], axis=0)
            test_label = np.concatenate([test_label, label], axis=0)
        print('test ' + str(i))
        predicted_label = model.predict(test_data)
        predicted_prob = model.predict_proba(test_data)
        acc[i], Sn[i], Sp[i], F1[i] = metric(y_true=test_label, y_pred=predicted_label)
        AF_nums[i, :], AF_dect_rate[i, :] = metrics.piece_detect_rate(
            predict_label=predicted_label, raw_label=label, time=time)
        try:
            AUC[i] = sklearn.metrics.roc_auc_score(y_true=test_label, y_score=predicted_prob[:, 1])
        except ValueError:
            AUC[i] = 0
        print("AF_nums=", AF_nums)
        print("AF_dect_rate=", AF_dect_rate)
        print("acc=", acc)
        print("Sn=", Sn)
        print("Sp=", Sp)
        print("F1=", F1)

    acc = acc[:, np.newaxis]
    Sn = Sn[:, np.newaxis]
    Sp = Sp[:, np.newaxis]
    AUC = AUC[:, np.newaxis]
    result = np.concatenate([acc, Sn, Sp, AUC], axis=1)
    a = pd.DataFrame([test_files, result])
    a.to_csv(path + 'result fold' + str(j) + '.csv')
    print("fold" + str(j) + " finish")
    AF_dect_rate_result = np.concatenate([AF_nums, AF_dect_rate], axis=1)
    a = pd.DataFrame(AF_dect_rate_result)
    a.to_csv(path + 'AF_nums and dect_rate.csv')

    predicted_label_ = predicted_label[:, np.newaxis]
    test_label_ = test_label[:, np.newaxis]
    predict_and_label = np.concatenate([predicted_label_, test_label_], axis=1)
    if total_label == []:
        total_label = predict_and_label
    else:
        total_label = np.concatenate([total_label, predict_and_label], axis=0)

np.save(path + 'total_label.npy', total_label)

acc, Sn, Sp, F1 = metric(y_true=total_label[:, 1], y_pred=total_label[:, 0])
print('total acc=', acc)
print('total Sn=', Sn)
print('total Sp=', Sp)
