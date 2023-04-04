########################################
# Wavelet Transform
# Result not good !!!!!
########################################
import numpy as np
from scipy import signal
from tqdm import trange
from multiprocessing import cpu_count
import datetime
from utility import dataset_loading

from sklearn.neighbors import KNeighborsClassifier


def dsp_processing(X_matrix):
    dsp_list = []
    for i in trange(0, X_matrix.shape[0]):
        sos = signal.butter(10, 100, 'hp', fs=1000, output='sos')
        dsp_list_temp = signal.sosfilt(sos, X_matrix[i,:])       
        for j in range(0, len(dsp_list_temp)):
            dsp_list_temp[j] = abs(dsp_list_temp[j])
        dsp_list.append(dsp_list_temp)
    dsp_list = np.array(dsp_list)
    print("Succeed !", end=" ")
    print("Shape =", dsp_list.shape)
    return dsp_list


if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = dataset_loading()
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data processing time: ', (end - start).seconds, "s")
    
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    dsp_list_train = dsp_processing(X_train)
    print("Label shape:", y_train.shape)  
    print("Start processing testing data")
    dsp_list_test = dsp_processing(X_test)
    print("Label shape:", y_test.shape)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    print("Start training (kNN)")
    start = datetime.datetime.now()
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(dsp_list_train, y_train)
    acc = model.score(dsp_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('KNN time: ', (end - start).seconds, "s")
    print("--------------------------------------")