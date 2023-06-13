import numpy as np
import antropy as ent
import math
import EntropyHub as EnH
import nolds


def fold_filenames(fold_index):
    if fold_index == 1:
        filenames = ['04015.csv', '04043.csv', '04048.csv', '04126.csv', '04746.csv', '04908.csv', '04936.csv',
                     '05091.csv', '05121.csv', '05261.csv', '06426.csv', '06453.csv', '06995.csv', '07162.csv',
                     '07859.csv', '07879.csv',
                     '07910.csv', '08434.csv', '08455.csv', '08219.csv', '08215.csv', '08378.csv', '08405.csv']
    if fold_index == 2:
        filenames = ['07910.csv', '08434.csv', '08455.csv', '08219.csv', '08215.csv', '08378.csv', '08405.csv',
                     '04015.csv', '04043.csv', '04048.csv', '04126.csv', '04746.csv', '04908.csv', '04936.csv',
                     '05091.csv', '05121.csv',
                     '05261.csv', '06426.csv', '06453.csv', '06995.csv', '07162.csv', '07859.csv', '07879.csv'
                     ]
    if fold_index == 3:
        filenames = ['05261.csv', '06426.csv', '06453.csv', '06995.csv', '07162.csv', '07859.csv', '07879.csv',
                     '07910.csv', '08434.csv', '08455.csv', '08219.csv', '08215.csv', '08378.csv', '08405.csv',
                     '04015.csv', '04043.csv',
                     '04048.csv', '04126.csv', '04746.csv', '04908.csv', '04936.csv', '05091.csv', '05121.csv'
                     ]
    if fold_index == 4:
        filenames = ['04048.csv', '04126.csv', '04746.csv', '04908.csv', '04936.csv', '05091.csv', '05121.csv',
                     '05261.csv', '06426.csv', '06453.csv', '06995.csv', '07162.csv', '07859.csv', '07879.csv',
                     '07910.csv', '08434.csv',
                     '08455.csv', '08219.csv', '08215.csv', '08378.csv', '08405.csv', '04015.csv', '04043.csv',
                     ]
    return filenames


def feature_extraction(RRI, label):
    # total RRI input and return feature_series of (L,feature_num)
    window_size = 30  # 30 RRI
    moving_size = 15
    feature_num = 4
    L = RRI.shape[0]
    ROI_RRI_num = int(np.floor(L / moving_size - 1))
    feature_series = np.zeros((ROI_RRI_num, feature_num))
    label_series = np.zeros((ROI_RRI_num))

    RRI = pulse_rejection_filter(RRI=RRI)

    for i in range(ROI_RRI_num):
        windowed_RRI = RRI[i * moving_size:i * moving_size + window_size]
        feature_set = windowed_RRI_feature_extraction(windowed_RRI=windowed_RRI)
        feature_series[i, :] = feature_set

        windowed_label = label[i * moving_size:i * moving_size + window_size]
        label_series[i] = 2 * np.sum(windowed_label) > window_size
    return feature_series, label_series


def windowed_RRI_feature_extraction(windowed_RRI):
    kmax = 10
    kbins = 200
    x = 800

    # Dk=FD_katz(RRI=windowed_RRI)
    # Dh=FD_Higuchi(RRI=windowed_RRI,kmax=kmax)
    # lle=LLE(RRI=windowed_RRI)
    # shen=shannon_entropy(RRI=windowed_RRI,k=kbins)
    cosen = CosEn(RRI=windowed_RRI)
    # rmssd=RMSSD(RRI=windowed_RRI)
    pnnx = pNNx(RRI=windowed_RRI, x=x)
    mad = MAD(RRI=windowed_RRI)
    # cov=CoV(RRI=windowed_RRI)
    mnn = MNN(RRI=windowed_RRI)

    # feature_set=np.asarray([[Dk,Dh,lle,shen,cosen,rmssd,pnnx,mad,cov,mnn]])
    feature_set = np.asarray([[cosen, pnnx, mad, mnn]])
    return feature_set


def pulse_rejection_filter(RRI):
    length = RRI.shape[0]
    for i in range(length):
        if 1 <= i & i < length - 1:
            RRI[i] = np.median(RRI[i - 1:i + 1])
    return RRI


# FD_K tools
def FD_katz(RRI):
    return ent.katz_fd(RRI)


# FD_H tools
def FD_Higuchi(RRI, kmax):
    return ent.higuchi_fd(RRI, kmax)


def Lk_mean(RRI, k):
    # full RRI series input
    M = RRI.shape[0]
    Lk_mean = 0
    for j in range(k):
        l = np.floor((M - j) / k)
        Ljk = 0
        for i in range(l):
            Ljk = Ljk + abs(RRI[j + i * k] - RRI[j + (i - 1) * k])
        NF = (M - 1) / (l * k)
        Ljk = Ljk / NF
        Lk_mean = Lk_mean + Ljk
    return Lk_mean


def FD_Higuchi_(RRI):
    k_max = 10
    Lk = np.zeros(k_max)
    for i in range(k_max):
        k = i + 1
        Lk[i] = Lk_mean(RRI, k)
    k = np.arange(k_max) + 1
    DH = -np.polyfit(x=np.log(k), y=np.log(Lk), deg=1)[0]
    return DH


# LLE tools
def LLE(RRI):
    T = int(np.floor(1 / np.median(abs(np.fft.fft(RRI)))))
    return nolds.lyap_r(data=RRI, emb_dim=5, lag=1, min_tsep=T)


# ShEn tools
def shannon_entropy(RRI, k):
    numofx = RRI.shape[0]
    maxV = np.max(RRI)
    minV = np.min(RRI)
    bin = np.linspace(minV, maxV, k + 1)

    bin_numx = [0] * k

    for x in RRI:
        for i in range(k):
            if x <= bin[i]:
                bin_numx[i - 1] += 1

    shannon_ent = 0

    for i in bin_numx:
        shannon_ent -= (i / numofx) * math.log((i / numofx), 2)

    return shannon_ent


# def shannon_entropy(RRI):
#     return nolds.sampen(data=RRI)


# CosEn tools
def CosEn(RRI):
    r = 0.2 * np.std(RRI)
    SEn = EnH.SampEn(Sig=RRI, r=r)[0]
    SEn = SEn[0]
    QSEn = SEn + np.log(2 * r)
    CoSEn = QSEn - np.log(HR_mean(RRI))
    return CoSEn


def HR_mean(RRI):
    return np.mean(60000 / RRI)


# RMSSD tools
def RMSSD(RRI):
    length = RRI.shape[0]
    sum = 0
    for i in range(length - 1):
        sum = sum + (RRI[i + 1] - RRI[i]) ** 2
    return np.sqrt(sum / (length - 1))


# pNNx tools
def pNNx(RRI, x):
    return np.sum(RRI > x) / (RRI.shape[0])


# MAD tools
def MAD(RRI):
    return np.median(abs(RRI - np.median(RRI)))


# CoV tools
def CoV(RRI):
    return np.std(RRI) / np.mean(RRI)


# MNN tools
def MNN(RRI):
    return np.mean(RRI)
