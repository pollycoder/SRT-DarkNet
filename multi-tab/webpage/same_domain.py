'''
Experiment for webpage fingerprinting to webpages with same home page domain
Here we referred to the research about webpage classification through frequency domain features
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from utility import dataset_loading
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
    print("Training shape =", fft_list.shape)
    return fft_list

    
if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)
    X_train, y_train, X_test, y_test = dataset_loading()
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    
    print("======================================")
    print("Start processing training data:")
    fft_list_train = fft_processing(X_train)
    print("Label shape:", y_train.shape)
    print("Start processing testing data")
    fft_list_test = fft_processing(X_test)
    print("Label shape:", y_test.shape)
    print("======================================")

    print("Start training (kNN)")
    model = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")

    print("Start training (RF)")
    model = RandomForestClassifier()
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")

    print("Start training (DT)")
    model = DecisionTreeClassifier()
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")

    print("Start training (LR)")
    model = LogisticRegression(penalty='l2')
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")

    print("Start training (SVM)")
    model = SVC(kernel='rbf', probability=True)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")

    print("Start training (NB)")
    model = MultinomialNB(alpha=0.01)
    model.fit(fft_list_train, y_train)
    acc = model.score(fft_list_test, y_test)
    print("Acc =", acc)
    print("--------------------------------------")