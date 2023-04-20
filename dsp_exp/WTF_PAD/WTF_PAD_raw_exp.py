import sys
sys.path.append("../")

from tools.data_loading import dataset
from tools.classifiers import DNN
from multiprocessing import cpu_count

'''
Experiment for non-frequency domain analysis
Script for experiment - WTF-PAD
Classifier: MLP
DSP: filterers - none, butter, gaussian
Output: accuracy
'''

fs = 1000
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("Raw training and testing for traces - WTF_PAD")
    cores = cpu_count()
    print("CPU cores:", cores)

    # Loading data
    X_train, y_train, X_test, y_test, X_valid, y_valid = \
    dataset(db_name="WTF_PAD", spec='none', filter='none')

    # Testing
    y_pred, acc = DNN(X_train, y_train, X_test, y_test)