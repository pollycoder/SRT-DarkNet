import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.dsp import spectrum
from tools.classifiers import DNN
import datetime
from multiprocessing import cpu_count
import numpy as np

'''
Experiment for frequency domain analysis
Script for experiment - Front
Classifier: MLP
DSP: filterers - none, butter, gaussian
Output: accuracy
'''

db_name = "Front"
print("Training and testing for traces - {}".format(db_name))
cores = cpu_count()
print("CPU cores:", cores)
print("--------------------------------------")


# Raw test
# Loading data
data_dir = "../npz/"
data_X = np.load(data_dir + db_name + "_X_raw.npz")
data_y = np.load(data_dir + db_name + "_y.npz")
train_X = data_X["train"]
test_X = data_X["test"]
train_y = data_y["train"]
test_y = data_y["test"]

# Testing
print("Testing - Raw {}".format(db_name))
y_pred, acc = DNN(train_X, train_y, test_X, test_y)
print("--------------------------------------")


# Freq test
# Loading data
data_dir = "../npz/"
data_X = np.load(data_dir + db_name + "_X_freq.npz")
data_y = np.load(data_dir + db_name + "_y.npz")
train_X = data_X["train"]
test_X = data_X["test"]
train_y = data_y["train"]
test_y = data_y["test"]

# Testing
print("Testing - Frequency {}".format(db_name))
y_pred, acc = DNN(train_X, train_y, test_X, test_y)
print("--------------------------------------")


# PSD test
# Loading data
data_dir = "../npz/"
data_X = np.load(data_dir + db_name + "_X_freq.npz")
data_y = np.load(data_dir + db_name + "_y.npz")
train_X = data_X["train"]
test_X = data_X["test"]
train_y = data_y["train"]
test_y = data_y["test"]

# Testing
print("Testing - Power {}".format(db_name))
y_pred, acc = DNN(train_X, train_y, test_X, test_y)
print("{}: All tests finished ! ".format(db_name))
