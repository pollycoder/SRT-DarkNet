###################################################
# Experiment for webpage fingerprinting to webpages 
# with same home page domain;
# Here we referred to the research about webpage 
# classification through frequency domain features.
# Exp: Power spectrum (Self-adapt)
###################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from utility import dataset_loading_wt,showScatter
import matplotlib.pyplot as plt

import datetime
import random
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


def double_fft(X_matrix):
    fft_list = fft_processing(X_matrix)
    fft_list = fft_processing(fft_list)
    return fft_list


if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = dataset_loading_wt()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = X_train, y_train, X_test, y_test
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data processing time: ', (end - start).seconds, "s")
    
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    fft_list_train = fft_processing(X_train)
    print("Label shape:", y_train.shape)  
    print("Start processing testing data")
    fft_list_test = fft_processing(X_test)
    print("Label shape:", y_test.shape)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")


    print("Start training (kNN)")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(fft_list_train, y_train)
    print("Training succeeded !")
    print("Now start testing...")
    start = datetime.datetime.now()
    y_pred = model.predict(fft_list_test)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('KNN time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    

    # Plotting
    print("Start plotting...")
    n = 5
    max = 80
    random_list = random.sample(range(1,76),n)
    print("Random web: ", random_list)
    X_plot_train = []
    y_plot_train = []
    X_plot_raw = []
    for i in trange(0, fft_list_train.shape[0]):
        if y_train[i] in random_list:
            x_list = fft_list_train[i,:].tolist()
            x_raw = X_train_raw[i,:].tolist()
            X_plot_train.append(x_list)
            y_plot_train.append(y_train[i])
            X_plot_raw.append(x_raw)
    X_plot_train = np.array(X_plot_train)
    y_plot_train = np.array(y_plot_train)
    X_plot_raw = np.array(X_plot_raw)
            

    X_plot_test = []
    y_plot_test = []
    y_plot_pred = []
    X_plot_rawtest = []
    for i in trange(0, fft_list_test.shape[0]):
        if y_test[i] in random_list:
            x_list = fft_list_test[i,:].tolist()
            x_rawtest = X_test_raw[i,:].tolist()
            X_plot_test.append(x_list)
            X_plot_rawtest.append(x_rawtest)
            y_plot_test.append(y_test[i])
            y_plot_pred.append(y_pred[i])
    X_plot_test = np.array(X_plot_test)
    y_plot_test = np.array(y_plot_test)
    X_plot_rawtest = np.array(X_plot_rawtest)

            
    print("X_plot_train shape: ", X_plot_train.shape)
    print("y_plot_train shape: ", y_plot_train.shape)
    print("X_plot_test shape: ", X_plot_test.shape)
    print("y_plot_test shape: ", y_plot_test.shape)
    print("X_plot_raw shape: ", X_plot_raw.shape)
    print("X_plot_rawtest shape: ", X_plot_rawtest.shape)

    showScatter(X_plot_train, y_plot_train, X_plot_test, y_plot_test, "Result-PowerSpec-WalkieTalkie", acc, 1, n, max)
    showScatter(X_plot_raw, y_plot_train, X_plot_rawtest, y_plot_test, "Result-Raw-WalikieTalkie", 0.0338, 2, n, max)
    plt.show()