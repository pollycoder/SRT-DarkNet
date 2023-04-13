import pickle as pkl
import numpy as np
import datetime
from tqdm import trange
from scipy.interpolate import interp1d

dataset_dir = "../../defense_datasets/"

############################################
# Data loading module
# WTF-PAD, Front, DF
# Loading the datasets and resample the data
############################################

# Resample the data
def resample(time, signal, resample_rate, cutoff_time):
    # Create a new time sequence, and find the ending time
    index = 0
    repeat_list = []
    while True:
        if time[index] == 0 or index == len(time) - 1:
            break
        if time[index] == time[index + 1] and time[index] != 0:
            repeat_list.append(index)
        index += 1
    time = time[0:index]
    time = np.delete(time, repeat_list)
    signal = signal[0:index]
    signal = np.delete(signal, repeat_list)
    new_time = np.arange(time[0], time[-1], 1.0 / resample_rate)

    # Interpolation for resampling
    f = interp1d(time, signal, kind='cubic')
    new_signal = f(new_time)

    # Padding or cutting the signal to make it have the length
    # Length = cutoff time * sampling rate
    length = cutoff_time * resample_rate
    while len(new_signal) < length:
        zero_list = np.zeros(length - len(new_signal)).tolist()
        new_signal = np.append(new_signal, zero_list)
    if len(new_signal) > length:
        new_signal = new_signal[0:length]
    return new_signal


# Loading dataset
def data_processing(prop, db_name):
    start = datetime.datetime.now()
    print("========================================")
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
    print("Direction min:", np.min(direction))
    print("Label shape:", label.shape)
    end = datetime.datetime.now()
    print('Data loading time: ', (end - start).seconds, "s")
    return time, direction, label


