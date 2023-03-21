'''
Experiment for webpage fingerprinting to webpages with same home page domain
Here we referred to the research about webpage classification through frequency domain features
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utility import dataset_loading

import datetime
import numpy as np
from tqdm import trange
from multiprocessing import cpu_count

def fft_processing(X_matrix):
    fft_list = []
    for i in trange(0, X_matrix.shape[0]):
        fft_list_temp = np.fft.rfft(X_matrix[i,:]).tolist()
        for j in range(0, len(fft_list_temp)):
            fft_list_temp[j] = abs(fft_list_temp[j])
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
    fft_list_train = X_train
    print("Label shape:", y_train.shape)  
    print("Start processing testing data")
    fft_list_test = X_test
    print("Label shape:", y_test.shape)
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    print("Start training (LD)")
    start = datetime.datetime.now()
    model = LinearDiscriminantAnalysis()
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('LD time: ', (end - start).seconds, "s")
    print("--------------------------------------")


    print("Start training (kNN)")
    start = datetime.datetime.now()
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('KNN time: ', (end - start).seconds, "s")
    print("--------------------------------------")

    print("Start training (RF)")
    start = datetime.datetime.now()
    model = RandomForestClassifier(n_jobs=-1, verbose=1)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('RF time: ', (end - start).seconds, "s")
    print("--------------------------------------")

    print("Start training (DT)")
    start = datetime.datetime.now()
    model = DecisionTreeClassifier()
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('DT time: ', (end - start).seconds, "s")
    print("--------------------------------------")

    print("Start training (LR)")
    start = datetime.datetime.now()
    model = LogisticRegression(n_jobs=-1, solver='lbfgs', max_iter=3000)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('LR time: ', (end - start).seconds, "s")
    print("--------------------------------------")

    print("Start training (SVM)")
    start = datetime.datetime.now()
    model = SVC(kernel='rbf', probability=True)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('SVM time: ', (end - start).seconds, "s")
    print("--------------------------------------")

   