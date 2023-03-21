import pickle as pkl
import numpy as np


def dataset_loading():
    print("Loading datasets ....")
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_NoDef.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_NoDef.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_NoDef.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_test_NoDef.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


def dataset_loading_multitab():
    print("Loading datasets...")
    dataset_dir = "../../datasets/1180filter/"
    with open(dataset_dir + "classifier.pickle", 'rb') as handle:
        train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "test.pickle", 'rb') as handle:
        test = np.array(pkl.load(handle, encoding='latin1')) 
    X_train = train[:, 0:20000]
    y_train = train[:, 20000]
    X_test = test[:, 0:20000]
    y_test = test[:, 20000]
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


