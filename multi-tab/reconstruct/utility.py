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


