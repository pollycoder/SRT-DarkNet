import numpy as np
from tqdm import trange
from scipy.signal import lfilter, butter, firwin

#########################################
# Digital signals processing module
# PSD: Get the power spectrum of the trace
# Filterers: low-pass - gaussian + butter
#########################################

# Get power spectrum density
def psd(X_matrix, filter='none'):
    fft_list = []
    for i in trange(0, X_matrix.shape[0]):
        signal = X_matrix[i,:]
        if filter == 'none':
            continue
        elif filter == 'gaussian':
            signal = gaussian_lowpass_filter(signal, sample_rate=1000, cutoff_freq=30, std=5)
        elif filter == 'butter':
            signal = butter_lowpass_filter(signal, cutoff_freq=30, fs=1000, order=5)
        else:
            try:
                print("Filter name error: The filterer doesn't exist !")
            except ValueError:
                print("Filter name should be 'str' type")
        fft_res = np.fft.fft(signal)
        fft_res = abs(fft_res)[:len(fft_res)//2] / len(signal) * 2

        corr = np.correlate(signal,signal,"same")
        corr_fft = np.fft.fft(corr)
        psd_corr_res = np.abs(corr_fft)[:len(fft_res)] / len(signal) * 2
        fft_list_temp = psd_corr_res.tolist()
        fft_list.append(fft_list_temp)
    fft_list = np.array(fft_list)
    print("Succeed !", end=" ")
    print("Shape =", fft_list.shape)
    return fft_list

# Butter Filterer
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Gaussian Filterer
def gaussian_lowpass_filter(signal, sample_rate, cutoff_freq, std):
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    num_taps = 101
    b = firwin(num_taps, normalized_cutoff_freq, window='gaussian', width=std)
    filtered_signal = lfilter(b, 1, signal)
    return filtered_signal