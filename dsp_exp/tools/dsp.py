import numpy as np
from tqdm import trange
from scipy.signal import lfilter, butter, firwin
from scipy.fftpack import fft, ifft
import datetime

'''
Digital signals processing module
    1. Resampling: fill in loads of 0-s
    2. Spectrums: Get the power spectrum of the trace
    3. Filter: filt the signals
'''

#####################################################################
# Resampling the signals
# Method: insert zeros into the  array to recover the original signal
#####################################################################

# Insert seros
def adaptive_resample(timestamps, signal, fs):
    interval = 1 / fs
    signal_list = []
    signal_list.append(signal[0])
    for i in range(0, len(timestamps) - 1):
        timestamp = timestamps[i]
        while True:
            signal_list.append(0)
            timestamp += interval
            if timestamp >= timestamps[i + 1]:
                break
        signal_list.append(signal[i + 1])
    new_signal = np.array(signal_list)
    return new_signal

def resample(time, signal, resample_rate, cutoff_time):
    # Create a new time sequence, and find the ending time
    # Remove the repeating timestamps
    index = 0
    repeat_list = []
    while True:
        if time[index] == 0 or index == len(time) - 1:
            break
        if time[index] == time[index + 1] and time[index] != 0:
            repeat_list.append(index)
        index += 1
    time = time[0:index]
    time = np.delete(time, repeat_list)
    signal = signal[0:index]
    signal = np.delete(signal, repeat_list)

    # Interpolation for resampling
    new_signal = adaptive_resample(time, signal, fs=resample_rate)
    

    # Padding or cutting the signal to make it have the length
    # Length = cutoff time * sampling rate
    length = cutoff_time * resample_rate
    while len(new_signal) < length:
        zero_list = np.zeros(length - len(new_signal)).tolist()
        new_signal = np.append(new_signal, zero_list)
    if len(new_signal) > length:
        new_signal = new_signal[0:length]
    return new_signal

##########################################
# Get different spectrums for analysis
# Spectrums:
#   Frequency spectrum
#   Power spectrum
##########################################
def spectrum(X_matrix, filter='none', spec='ps-corr'):
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
    elif filter == 'kalman':
        print("Filterer: Kalman")
    else:
        print("Filterer: Unknown")

    if spec == 'ps-corr':
        print("Spectrum: Power Spectrum with Correlation")
    elif spec == 'freq':
        print("Spectrum: Frequency Spectrum")

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
            signal = butter_lowpass_filter(signal, cutoff_freq=10, fs=300, order=5)
        elif filter == 'kalman':
            signal = kalman_filter(signal)
        else:
            try:
                print("Filter name error: The filterer doesn't exist !")
            except ValueError:
                print("Filter name should be 'str' type")
        fft_res = np.fft.fft(signal)
        fft_res = abs(fft_res)[:len(fft_res)//2] / len(signal) * 2

        if spec == 'ps-corr':
            corr = np.correlate(signal,signal,"same")
            corr_fft = np.fft.fft(corr)
            psd_corr_res = np.abs(corr_fft)[:len(fft_res)] / len(signal) * 2
            fft_list_temp = psd_corr_res.tolist()
        elif spec == 'freq':
            fft_list_temp = fft_res

        fft_list.append(fft_list_temp)
    fft_list = np.array(fft_list)
    end = datetime.datetime.now()
    print("Succeed !", end=" ")
    print('Spectrum generating time: ', (end - start).seconds, "s")
    return fft_list

######################################
# Filters:
# Butterworth - lowpass, highpass
# Gaussian - lowpass
# Directly cutting - low
# Kalman
######################################

# Butter Filter - lowpass
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Gaussian Filter
def gaussian_lowpass_filter(signal, sample_rate, cutoff_freq, std):
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    num_taps = 101
    b = firwin(num_taps, normalized_cutoff_freq, window='gaussian', width=std)
    filtered_signal = lfilter(b, 1, signal)
    return filtered_signal


# Butter Filter - highpass
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_highpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Cutting filter - lowpass
# Directly cut off high
def direct_lowpass_filter(signal, cutoff_freq, fs):
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(len(signal_fft), 1/fs)
    signal_fft[freqs > cutoff_freq] = 0
    signal_filtered = np.real(ifft(signal_fft))
    return signal_filtered


# Kalman filter
def kalman_filter(signal, Q=1E-4, R=1E-5):
    x_hat = np.zeros(len(signal))   
    P = np.zeros(len(signal))       
    K = np.zeros(len(signal))       
    x_hat[0] = signal[0]
    P[0] = 1.0
    for k in range(1, len(signal)):
        x_hat[k] = x_hat[k-1]
        P[k] = P[k-1] + Q
        K[k] = P[k] / (P[k] + R)
        x_hat[k] = x_hat[k] + K[k] * (signal[k] - x_hat[k])
        P[k] = (1 - K[k]) * P[k]      
    return x_hat
