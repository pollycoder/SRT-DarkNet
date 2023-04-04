###################################################
# Experiment for webpage fingerprinting to webpages 
# with same home page domain;
# Here we referred to the research about webpage 
# classification through frequency domain features.
# Exp: Cepstrum
###################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from utility import dataset_loading, showScatter

import datetime
import numpy as np
from tqdm import trange
from multiprocessing import cpu_count

def fft_processing(X_matrix):
    fft_list = []
    for i in trange(0, X_matrix.shape[0]):
        signal = X_matrix[i,:]
        fft_res = np.fft.fft(signal)  
        ceps_res = np.fft.ifft(np.log(fft_res + 1e-5)).real  # To avoid ln(0)
        ceps_res = np.abs(ceps_res)[:len(signal)//2]
        fft_list_temp = ceps_res.tolist()
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
    fft_list_train = fft_processing(X_train)
    print("Label shape:", y_train.shape)  
    print("Start processing testing data")
    fft_list_test = fft_processing(X_test)
    print("Label shape:", y_test.shape)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")


    print("Start training (kNN)")
    model = KNeighborsClassifier(n_neighbors=100)
    model.fit(fft_list_train, y_train)
    print("Training succeeded !")
    print("Now start testing...")
    start = datetime.datetime.now()
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('KNN time: ', (end - start).seconds, "s")
    print("--------------------------------------")

    '''
    print("Start training (RF)")
    model = RandomForestClassifier(n_jobs=-1, verbose=1, n_estimators=2000)
    model.fit(fft_list_train, y_train)
    print("Training succeeded !")
    print("Now start testing...")
    start = datetime.datetime.now()
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('RF time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    '''

    # Plot
    showScatter(fft_list_train, y_train, fft_list_test, y_test, "Result", acc, 1)