import sys
sys.path.append("../")
from tools.data_loading import dataset_loading_nodef
from tools.plotting import rgb
from tools.dsp import psd
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

##########################################
# Experiment for frequency domain analysis
# Script for RGB (single website) - No denfense
# DSP: filterers - none, butter, gaussian
# Output: RGB
##########################################

if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    X_train, y_train, X_test, y_test = dataset_loading_nodef()
    X_train = X_train = psd(X_train, filter='butter')
    

    min = 10
    max = 40
    width = 50
    title = "NoDef-RGB"
    rgb(X_train, y_train, min, max, width, title)
    plt.show()

   
