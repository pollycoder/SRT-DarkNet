import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.dsp import spectrum
from tools.classifiers import DNN
import datetime
from multiprocessing import cpu_count

'''
Experiment for frequency domain analysis
Script for experiment - WTF-PAD
Classifier: MLP
DSP: filterers - none, butter, gaussian
Output: accuracy
'''

fs = 250
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("PSD Training and testing for traces - WTF-PAD")
    cores = cpu_count()
    print("CPU cores:", cores)
    print("======================================")

    # Loading data
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset(db_name="WTF_PAD")

    # Testing
    y_pred, acc = DNN(X_train, y_train, X_test, y_test)