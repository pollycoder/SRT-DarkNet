import sys
sys.path.append("../")

from tools.data_loading import data_processing, data_direction
from tools.plotting import showScatter, sample_scatter
from tools.dsp import psd
from tools.classifiers import RF, DNN, DT, LR

import matplotlib.pyplot as plt
import datetime
from multiprocessing import cpu_count

##########################################
# Experiment for frequency domain analysis
# Script for experiment - DF
# Classifier: MLP
# DSP: filterers - none, butter, gaussian
# Output: accuracy and scatter plot
##########################################

fs = 1000
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("Raw training and testing for traces - DF")
    cores = cpu_count()
    print("CPU cores:", cores)

    # Loading data
    print("Start full-data experiment...")
    X_train, y_train, X_test, y_test = data_processing(prop=0.1, db_name="DF")
    
    # Processing data
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    fft_list_train = X_train                          # Change the filterer
    print("Start processing testing data")
    fft_list_test = X_test                            # Change the filterer
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    # Testing
    y_pred, acc = DNN(fft_list_train, y_train, fft_list_test, y_test)