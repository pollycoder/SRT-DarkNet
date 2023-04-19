import pickle as pkl
import numpy as np
import datetime
from tqdm import trange
from tools.dsp import resample

dataset_dir = "../../defense_datasets/"

'''
Data loading module
 Datasets: WTF-PAD, Front, DF
 Loading the datasets
'''

# Loading dataset
# Timestamps and directions
def data_processing(prop, db_name):
    start = datetime.datetime.now()
    print("Process the dataset and resample it")
    time, direction, label = data_loading(db_name)
    print("-----------------------------------------")
    print("Resampling signals")
    signal_set = []
    for i in trange(0, time.shape[0]):
        signal = direction[i,:]
        timestamps = time[i,:]
        signal = resample(timestamps, signal, resample_rate=250, cutoff_time=20)
        signal = signal.tolist()
        signal_set.append(signal)
    signal_set = np.array(signal_set)
    print("Succeeded.")
    print("New signal shape:", signal_set.shape)

    sample_num = int(signal_set.shape[0] * (1-prop))
    X_train = signal_set[0:sample_num,:]
    y_train = label[0:sample_num]
    X_test = signal_set[sample_num:-1,:]
    y_test = label[sample_num:-1]

    end = datetime.datetime.now()
    print('Total data processing time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test


# Loading dataset
# Only directions
def data_direction(prop, db_name):
    start = datetime.datetime.now()
    print("Process the dataset")
    time, direction, label = data_loading(db_name)
    print("-----------------------------------------")
    sample_num = int(direction.shape[0] * (1-prop))
    X_train = direction[0:sample_num,:]
    y_train = label[0:sample_num]
    X_test = direction[sample_num:-1,:]
    y_test = label[sample_num:-1]
    end = datetime.datetime.now()
    print('Total data processing time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test


# Loading dataset
def data_loading(dataset):
    start = datetime.datetime.now()
    print("Loading datasets")
    data = np.load(dataset_dir + dataset + ".npz")
    time = data["time"]
    direction = data["direction"]
    label = data["label"]
    print("Succeeded. Dataset:", dataset)
    print("Time stamps shape:", time.shape)
    print("Directions shape:", direction.shape)
    print("Label shape:", label.shape)
    end = datetime.datetime.now()
    print('Data loading time: ', (end - start).seconds, "s")
    return time, direction, label


