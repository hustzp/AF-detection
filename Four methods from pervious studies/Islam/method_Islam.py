import pandas as pd
import numpy as np
import os



xr = 800  # xr is a constant ，equal to the duration of an average heartbeat in NSR，normally as 800ms
epsilon = 32  # constant ε = 32ms
bin_length = 40  # length of bin is 40ms

def HBA(rri_data, threshold: float, window_size: int, step_size: int):

    
    #data = np.array([])

    num_windows = (len(rri_data) - window_size) // step_size + 1

    
    detected_rhythm_type = np.zeros(len(rri_data))

  
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

        
        xa = np.mean(hbd_data)
        xm = np.median(hbd_data)




        if abs(xa - xm) < epsilon:
            #window_type = 'single'
            W_select = hbd_data
        else:
            #window_type = 'mixed'
     
            w1 = hbd_data[hbd_data < xm]
            w2 = hbd_data[hbd_data >= xm]

       
            x1 = np.mean(w1)
            x2 = np.mean(w2)
            if abs(x1 - xm) < abs(x2 - xm):

                W_select = w1
            else:
   
                W_select = w2


        s = np.mean(W_select)
        if s>0:
            hbas_data = xr * (hbd_data - xa) / s
        else:
            continue


        window_duration = np.sum(np.abs(hbas_data))


        n_bins = max(int(window_duration // bin_length), 2)
        '''

        if n_bins <= 1:
            raise ValueError("Invalid number of bins (must be a positive integer)")
        '''
        hist, _ = np.histogram(hbas_data, bins=n_bins)
        hist = hist[hist > 0]
        nonzero_bins = np.count_nonzero(hist)
        p = hist / window_size
        entropy = -(np.sum(nonzero_bins) * np.sum(p * np.log2(p))) / window_size

        if entropy >= threshold:
            detected_rhythm_type[start:end] = 1.0  
        else:
            detected_rhythm_type[start:end] = 0.0

        '''
        if (i == 2):
            print("第{}个窗口中的数据：{}".format(i + 1, hbd_data))
            print(xa, xm, s, hbas_data)
            print(n_bins, hist, p, nonzero_bins, np.sum(p * np.log2(p)), entropy)
        '''

    #data = np.hstack([rri_data,detected_rhythm_type])
    detected_rhythm_type = detected_rhythm_type.reshape(-1)
    return detected_rhythm_type
