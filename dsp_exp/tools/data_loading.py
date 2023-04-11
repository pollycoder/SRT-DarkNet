import pickle as pkl
import numpy as np
import datetime


##########################
# Data loading module
# NoDef, WTF-PAD, W-T for DF
##########################

def dataset_loading_nodef():
    print("Loading datasets ....")
    start = datetime.datetime.now()
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_NoDef.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
        #X_train = X_train[:,1000:1100]
    with open(dataset_dir + "y_train_NoDef.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_NoDef.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
        #X_test = X_test[:,1000:1100]
    with open(dataset_dir + "y_test_NoDef.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data loading time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test


def dataset_loading_wtfpad():
    print("Loading datasets ....")
    start = datetime.datetime.now()
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_WTFPAD.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_WTFPAD.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_WTFPAD.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_test_WTFPAD.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    print("Data loaded successfully !")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data loading time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test


def dataset_loading_wt():
    print("Loading datasets ....")
    start = datetime.datetime.now()
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_WalkieTalkie.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_WalkieTalkie.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_WalkieTalkie.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_test_WalkieTalkie.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    print("Data loaded successfully !")
    print("Data loaded successfully !")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape: ", y_test.shape)
    end = datetime.datetime.now()
    print('Data loading time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test