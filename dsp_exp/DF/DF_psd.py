import sys
sys.path.append("../")
from tools.data_loading import data_processing
from tools.dsp import psd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count

##########################################
# Experiment for frequency domain analysis
# Script for drawing PSD - DF
# DSP: filterers - none, butter, gaussian
# Output: PSD
##########################################


if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    X_train, y_train, X_test, y_test = data_processing(prop=0.1, db_name="DF")
    X_train = psd(X_train, filter='butter-low')


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
    plt.suptitle("Nodef-PSD (n=" + str(n) + ")")
    plt.savefig("../result/psd/DF_exp.png")
   
