import pickle as pkl
import numpy as np
import datetime
from tqdm import trange
from tools.dsp import resample
from tools.dsp import spectrum

dataset_dir = "../../defense_datasets/"

'''
Data loading module
 Datasets: WTF-PAD, Front, DF
 Loading the datasets
'''

# Loading the final set
# Already processed data
def dataset(prop_test=0.1, prop_valid=0.2, db_name="DF", type="td", spec="ps-corr", filter="none"):

    # Data loading and preprocessing
    X_train, y_train, X_valid, y_valid, X_test, y_test = 0, 0, 0, 0, 0, 0
    if type == "td":
        X_train, y_train, X_test, y_test, X_valid, y_valid \
        = data_processing(prop_test, prop_valid, db_name)
    elif type == "d":
        X_train, y_train, X_test, y_test, X_valid, y_valid \
        = data_direction(prop_test, prop_valid, db_name)
    else:
        print("Type ERROR: Please input the correct data type: \
              td for dataset with time, d for direction-only dataset.")
        return
    print("------------------------------------------------------")
    
    # DSP
    print("Start generating training spectrums...")
    X_train = spectrum(X_matrix=X_train, filter=filter, spec=spec)
    print("Start generating validating spectrums...")
    X_valid = spectrum(X_matrix=X_valid, filter=filter, spec=spec)
    print("Start generating testing spectrums...")
    X_test = spectrum(X_matrix=X_test, filter=filter, spec=spec)

    return X_train, y_train, X_test, y_test, X_valid, y_valid





# Loading dataset
# Timestamps and directions
def data_processing(prop_test=0.1, prop_valid=0.2, db_name="DF"):
    start = datetime.datetime.now()
    print("Process the dataset and resample it")
    time, direction, label = data_loading(db_name)
    print("-----------------------------------------")
    print("Resampling signals")
    signal_set = []
    for i in trange(0, time.shape[0]):
        signal = direction[i,:]
        timestamps = time[i,:]
        signal = resample(timestamps, signal, 
                          resample_rate=250, 
                          cutoff_time=20)
        signal = signal.tolist()
        signal_set.append(signal)
    signal_set = np.array(signal_set)
    print("Succeeded.")
    print("New signal shape:", signal_set.shape)

    # Dividing the dataset
    total_num = signal_set.shape[0]
    sample_num = int(total_num * (1-prop_test-prop_valid))
    valid_num = int(total_num * prop_valid)
    test_num = int(total_num * prop_test)

    X_train = signal_set[0:sample_num,:]
    y_train = label[0:sample_num]
    X_valid = signal_set[sample_num:sample_num+valid_num, :]
    y_valid = label[sample_num:sample_num+valid_num]
    X_test = signal_set[total_num-test_num:-1,:]
    y_test = label[total_num-test_num:-1]

    end = datetime.datetime.now()
    print('Total data processing time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test, X_valid, y_valid


# Loading dataset
# Only directions
def data_direction(prop_test=0.1, prop_valid=0.2, db_name="DF"):
    start = datetime.datetime.now()
    print("Process the dataset")
    time, direction, label = data_loading(db_name)
    print("-----------------------------------------")
    # Dividing the dataset
    total_num = direction.shape[0]
    sample_num = int(total_num * (1-prop_test-prop_valid))
    valid_num = int(total_num * prop_valid)
    test_num = int(total_num * prop_test)
    
    X_train = direction[0:sample_num,:]
    y_train = label[0:sample_num]
    X_valid = direction[sample_num:sample_num+valid_num, :]
    y_valid = label[sample_num:sample_num+valid_num]
    X_test = direction[total_num-test_num:-1,:]
    y_test = label[total_num-test_num:-1]

    end = datetime.datetime.now()
    print('Total data processing time: ', (end - start).seconds, "s")
    return X_train, y_train, X_test, y_test, X_valid, y_valid


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


