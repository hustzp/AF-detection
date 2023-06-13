import pandas as pd
import numpy as np
import os


# 定义常量
xr = 800  # xr是一个常数，等于在NSR中一次平均心跳的持续时间，一般假定为0.8s，也即800ms
epsilon = 32  # 常数ε取32ms
bin_length = 40  # 每个箱子的宽度为40ms

def HBA(rri_data, threshold: float, window_size: int, step_size: int):

    #创立一个数组，用于存储后面需要返回的data数据，包括原始数据和经由算法检测到的标签
    #data = np.array([])

    num_windows = (len(rri_data) - window_size) // step_size + 1

    #定义空的数组用于存储检测结果
    detected_rhythm_type = np.zeros(len(rri_data))

    # 循环遍历每个窗口
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        hbd_data = rri_data[start:end]

        # 错误处理
        if np.isnan(hbd_data).any() or np.isinf(hbd_data).any():
            print("Error: Invalid data detected in window ", i)
            continue
        elif len(hbd_data) == 0:
            print("Error: Empty data detected in window ", i)
            continue

        # 计算平均值和中位数
        xa = np.mean(hbd_data)
        xm = np.median(hbd_data)


        #定义一个节律R，R可以是AF节律，也可以是NAF节律
        #定义一个窗口类型，可以是单一的，也可以是混合的
        if abs(xa - xm) < epsilon:
            #window_type = 'single'
            W_select = hbd_data
        else:
            #window_type = 'mixed'
            # 将HBD数据按照中位数分为两个子集
            w1 = hbd_data[hbd_data < xm]
            w2 = hbd_data[hbd_data >= xm]

            # 计算各自的平均值，选择更接近中位数的子集作为该窗口包含的数据内容
            x1 = np.mean(w1)
            x2 = np.mean(w2)
            if abs(x1 - xm) < abs(x2 - xm):
                #w1窗口代表的节律占主导，且近似为单一节律
                W_select = w1
            else:
                #w2窗口代表的节律占主导，且近似为单一节律
                W_select = w2

        # 对HBD进行归一化处理
        s = np.mean(W_select)
        if s>0:
            hbas_data = xr * (hbd_data - xa) / s
        else:
            continue

        # 计算标准化熵，判断窗口所属的节律类型
        window_duration = np.sum(np.abs(hbas_data))
        #考虑到划分bin的时候可能出现箱子个数为0或1的情况，前者因为使用n_bins做hist直方图必须为非负数，
        #后者因为当箱子个数为1时hist数组只包含一个值，无法计算熵
        n_bins = max(int(window_duration // bin_length), 2)
        '''
        #错误处理,有上面的取2就不需要下面的错误处理
        if n_bins <= 1:
            raise ValueError("Invalid number of bins (must be a positive integer)")
        '''
        hist, _ = np.histogram(hbas_data, bins=n_bins)
        hist = hist[hist > 0]
        nonzero_bins = np.count_nonzero(hist)
        p = hist / window_size
        entropy = -(np.sum(nonzero_bins) * np.sum(p * np.log2(p))) / window_size

        if entropy >= threshold:
            detected_rhythm_type[start:end] = 1.0   #这里是把这个窗口内所有的数据都划分到AF节律，也就是对应标签’1‘
        else:
            detected_rhythm_type[start:end] = 0.0

        '''# 打印第i+1个窗口中的数据
        if (i == 2):
            print("第{}个窗口中的数据：{}".format(i + 1, hbd_data))
            print(xa, xm, s, hbas_data)
            print(n_bins, hist, p, nonzero_bins, np.sum(p * np.log2(p)), entropy)
        '''
    #将检测结果加入到原始数据中
    #data = np.hstack([rri_data,detected_rhythm_type])
    detected_rhythm_type = detected_rhythm_type.reshape(-1)
    return detected_rhythm_type