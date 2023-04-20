import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.plotting import rgb
from tools.dsp import psd
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

##########################################
# Experiment for frequency domain analysis
# Script for RGB (single website) - DF
# DSP: filterers - none, butter, gaussian
# Output: RGB
##########################################

if __name__ == '__main__':  
    print("Frequency domain analysis attack")
    cores = cpu_count()
    print("CPU cores:", cores)

    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset(db_name="DF")
    
    min = 10
    max = 40
    width = 50
    title = "DF-RGB"
    rgb(X_train, y_train, min, max, width, title)
    plt.savefig("../reuslt/rgb/DF_exp.png")

   
