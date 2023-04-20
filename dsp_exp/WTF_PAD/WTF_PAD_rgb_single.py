import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.plotting import rgb_singlepage
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

    # Loading data
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset(db_name="WTF_PAD")
    

    index = 70
    width = 50
    title = "WTF_PAD-RGB(n=" + str(index) + ")"
    rgb_singlepage(X_train, y_train, index, width, title)
    plt.savefig("../result/rgb/WTF_PAD_raw_exp.png")