import sys
sys.path.append("../")
from tools.data_loading import data_processing
from tools.plotting import showScatter, sample_scatter
from tools.dsp import psd
from tools.classifiers import kNN
import matplotlib.pyplot as plt
import datetime
from multiprocessing import cpu_count

##########################################
# Experiment for frequency domain analysis
# Script for experiment - WTF-PAD
# Classifier: MLP
# DSP: filterers - none, butter, gaussian
# Output: accuracy and scatter plot
##########################################


fs = 1000
cutoff_freq = 30
order = 5

if __name__ == '__main__':  
    print("PSD Training and testing for traces - WTF-PAD")
    cores = cpu_count()
    print("CPU cores:", cores)

    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = data_processing(prop=0.1, db_name="WTF_PAD")
    
    #X_train_raw, y_train_raw, X_test_raw, y_test_raw = X_train, y_train, X_test, y_test
    
    print("======================================")
    print("Start processing training data:")
    start = datetime.datetime.now()
    fft_list_train = psd(X_train, filter='butter-low')                          # Change the filterer
    print("Start processing testing data")
    fft_list_test = psd(X_test, filter='butter-low')                            # Change the filterer
    end = datetime.datetime.now()
    print('Feature extracting time: ', (end - start).seconds, "s")
    print("======================================")

    
    y_pred, acc = kNN(fft_list_train, y_train, fft_list_test, y_test)
    
    
    
    # Scattering
    n = 10                                                                  # Classes going to plot
    X_plot_train, y_plot_train, \
    X_plot_test, y_plot_test,   \
    X_plot_raw, X_plot_rawtest = sample_scatter(fft_list_train, 
                                                y_train, fft_list_test, 
                                                y_test, y_pred, 
                                                X_train, 
                                                X_test, n)              # Choose the samples for scattering
    showScatter(X_plot_train, y_plot_train, 
                X_plot_test, y_plot_test, 
                "Result-PowerSpec-WTFPAD", acc, 1, n, max)
    showScatter(X_plot_raw, y_plot_train, 
                X_plot_rawtest, y_plot_test, 
                "Result-Raw", 0.23, 2, n, max)
    plt.show()