import sys
sys.path.append("../")
from tools.data_loading import data_processing
from tools.plotting import rgb_singlepage
from tools.dsp import psd
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

##########################################
# Experiment for frequency domain analysis
# Script for RGB (ALL websites) - WTF-PAD
# DSP: filterers - none, butter, gaussian
# Output: RGB
##########################################

if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    X_train, y_train, X_test, y_test = data_processing(prop=0.1, db_name="WTF_PAD")
    X_train = X_train = psd(X_train, filter='butter-low')
    

    index = 70
    width = 50
    title = "WTF_PAD-RGB(n=" + str(index) + ")"
    rgb_singlepage(X_train, y_train, index, width, title)
    plt.show()