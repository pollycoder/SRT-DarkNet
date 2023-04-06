###################################################
# Experiment for webpage fingerprinting to webpages 
# with same home page domain;
# Here we referred to the research about webpage 
# classification through frequency domain features.
# Exp: Power spectrum (Self-adapt)
###################################################


from utility import dataset_loading_wt,rgb
import datetime
import numpy as np
from tqdm import trange
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

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
    X_test = fft_processing(X_test)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    min = 71
    max = 94
    width = 30
    title = "Walkie-Talkie-RGB"
    rgb(X_train, y_train, min, max, width, title)
    plt.show()
    
    
   
