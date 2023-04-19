import sys
sys.path.append("../")
from tools.data_loading import data_processing
from tools.dsp import spectrum
from tools.classifiers import DNN
import datetime
from multiprocessing import cpu_count

##########################################
# Experiment for frequency domain analysis
# Script for experiment - Front
# Classifier: MLP
# DSP: filterers - none, butter, gaussian
# Output: accuracy
##########################################

fs = 1000
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("===========================================================")
    print("PSD Training and testing for traces - Front")
    cores = cpu_count()
    print("CPU cores:", cores)

    # Loading data
    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = data_processing(prop=0.1, db_name="Front")
    
    # Processing data
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    fft_list_train = spectrum(X_train, filter='butter-low')                          # Change the filterer
    print("Start processing testing data")
    fft_list_test = spectrum(X_test, filter='butter-low')                            # Change the filterer
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    # Testing
    y_pred, acc = DNN(fft_list_train, y_train, fft_list_test, y_test)
