import numpy as np
from tqdm import trange
from scipy.signal import lfilter, butter, firwin
from scipy.fftpack import fft, ifft
import datetime

#########################################
# Digital signals processing module
# PSD: Get the power spectrum of the trace
# Filterers: 
#   low-pass - gaussian + butter
#   high-pass - butter
#   combine: Window + Butter
#########################################

# Get power spectrum density
def psd(X_matrix, filter='none'):
    fft_list = []

    if filter == 'none':
        print("Filterer: None")
    elif filter == 'direct':
        print("Filterer: Direct Lowpass")
    elif filter == 'gaussian':
        print("Filterer: Gaussian Lowpass")
    elif filter == 'butter-low':
        print("Filterer: Butter Lowpass")
    elif filter == 'butter-high':
        print("Filterer: Butter Highpass")
    elif filter == 'window':
        print("Filterer: Window")
    elif filter == 'winb-low':
        print("Filterer: Window + Butter Lowpass")
    else:
        print("Filterer: Unknown")

    start = datetime.datetime.now()
    for i in trange(0, X_matrix.shape[0]):
        signal = X_matrix[i,:]
        if filter == 'none':
            signal = X_matrix[i,:]
        elif filter == 'direct':
            signal = direct_lowpass_filter(signal, cutoff_freq=30, fs=1000)
        elif filter == 'gaussian':
            signal = gaussian_lowpass_filter(signal, sample_rate=1000, cutoff_freq=30, std=5)
        elif filter == 'butter-low':
            signal = butter_lowpass_filter(signal, cutoff_freq=30, fs=1000, order=5)
        elif filter == 'butter-high':
            signal = butter_highpass_filter(signal, cutoff_freq=30, fs=1000, order=5)
        elif filter == 'window':
            window = np.hamming(len(signal))
            signal = window * signal
        elif filter == 'winb-low':
            window = np.hamming(len(signal))
            signal = window * signal
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
    end = datetime.datetime.now()
    print("Succeed !", end=" ")
    print('PSD generating time: ', (end - start).seconds, "s")
    return fft_list


# Butter Filterer - lowpass
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
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


# Butter Filterer - highpass
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_highpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Cutting filterer - lowpass
# Directly cut off high
def direct_lowpass_filter(signal, cutoff_freq, fs):
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(len(signal_fft), 1/fs)
    signal_fft[freqs > cutoff_freq] = 0
    signal_filtered = np.real(ifft(signal_fft))
    return signal_filtered
