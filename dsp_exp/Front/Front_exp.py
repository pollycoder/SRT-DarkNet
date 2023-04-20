import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.dsp import spectrum
from tools.classifiers import DNN
import datetime
from multiprocessing import cpu_count

'''
Experiment for frequency domain analysis
Script for experiment - Front
Classifier: MLP
DSP: filterers - none, butter, gaussian
Output: accuracy
'''

fs = 1000
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("===========================================================")
    print("PSD Training and testing for traces - Front")
    cores = cpu_count()
    print("CPU cores:", cores)

    # Loading data
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset(db_name="Front")

    # Testing
    y_pred, acc = DNN(X_train, y_train, X_test, y_test)
