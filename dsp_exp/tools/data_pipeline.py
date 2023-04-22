import sys
sys.path.append("../")
from data_loading import dataset
from dsp import spectrum
import numpy as np

data_dir = "../npz/"
dataset_dir = "../../defense_datasets/"

def data_saving(db_name):
    X_train, y_train, X_test, y_test, X_valid, y_valid = \
    dataset(db_name=db_name, type="d", spec='none', filter='none')
    np.savez(data_dir + "{}_X_raw".format(db_name), 
            train=X_train, test=X_test, valid=X_valid)
    np.savez(data_dir + "{}_y".format(db_name), 
            train=y_train, test=y_test, valid=y_valid)

    X_train_f = spectrum(X_train, filter='butter-low', spec='freq')
    X_test_f = spectrum(X_test, filter='butter-low', spec='freq')
    X_valid_f = spectrum(X_valid, filter='butter-low', spec='freq')
    np.savez(data_dir + "{}_X_freq".format(db_name), 
            train=X_train_f, test=X_test_f, valid=X_valid_f)

    X_train_f = spectrum(X_train, filter='butter-low', spec='ps-corr')
    X_test_f = spectrum(X_test, filter='butter-low', spec='ps-corr')
    X_valid_f = spectrum(X_valid, filter='butter-low', spec='ps-corr')
    np.savez(data_dir + "{}_X_ps".format(db_name), 
            train=X_train_f, test=X_test_f, valid=X_valid_f)
    

data_saving("DF")
data_saving("WTF_PAD")
data_saving("Front")









