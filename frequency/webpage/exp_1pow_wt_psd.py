###################################################
# Experiment for webpage fingerprinting to webpages 
# with same home page domain;
# Here we referred to the research about webpage 
# classification through frequency domain features.
# Exp: Power spectrum (Self-adapt)
###################################################

from utility import dataset_loading_wt
import matplotlib.pyplot as plt

import datetime
import numpy as np
from tqdm import trange
from multiprocessing import cpu_count

def fft_processing(X_matrix):
    fft_list = []
    for i in trange(0, X_matrix.shape[0]):
        signal = X_matrix[i,:]
        fft_res = np.fft.fft(signal)
        fft_res = abs(fft_res)[:len(fft_res)//2] / len(signal) * 2

        corr = np.correlate(signal,signal,"same")
        corr_fft = np.fft.fft(corr)
        psd_corr_res = np.abs(corr_fft)[:len(fft_res)] / len(signal) * 2
        fft_list_temp = psd_corr_res.tolist()
        fft_list.append(fft_list_temp)
    fft_list = np.array(fft_list)
    print("Succeed !", end=" ")
    print("Shape =", fft_list.shape)
    return fft_list


if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = dataset_loading_wt()
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data processing time: ', (end - start).seconds, "s")
    
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    X_train = fft_processing(X_train)
    print("Start processing testing data")
    fft_list_test = fft_processing(X_test)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")


    # Plot
    j = 0
    n = 54
    num = 5
    for i in range(0, X_train.shape[0]):
        if y_train[i] == n:
            j = j + 1
            plt.subplot(5, 1, j)
            y = range(1, X_train.shape[1] + 1, 1)
            plt.plot(y, X_train[i,:])
            maxrange = 300
            ti = np.arange(0, maxrange + 1, 10)
            plt.xlim(0, maxrange)  # 设定绘图范围
            plt.ylim(0, 20) 
            plt.xticks(ti)  # 设定刻度    
        if j == num:
            break
    plt.suptitle("WT-PSD (n=" + str(n) + ")")
    plt.show()
   
