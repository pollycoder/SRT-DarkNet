import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import orthogonal_mp

# 输入数据
timestamps = np.array([0, 1.5, 3.2, 4.7, 6.1, 7.6, 9.3])  # 时间戳
signal = np.array([1.2, 0, 3.5, 0, 0, 2.1, 0])  # 信号


def resample(timestamps, signal, fs):
    resampling_interval = int(1 / fs)  # 重采样间隔
    resampling_size = int((timestamps[-1] - timestamps[0]) / resampling_interval) + 1  # 重采样后样本数量

    # 构建稀疏矩阵
    rows = []
    cols = []
    data = []

    for i in range(resampling_size):
        t = i * resampling_interval + timestamps[0]
        for j in range(len(timestamps)):
            if abs(t - timestamps[j]) < 1e-10:
                rows.append(i)
                cols.append(j)
                data.append(1)
                break

    A = csr_matrix((data, (rows, cols)), shape=(resampling_size, len(timestamps)))

    # 使用OMP算法重构重采样信号
    resampled_signal = orthogonal_mp(A, signal, n_nonzero_coefs=None, tol=None, precompute=True)

    # 计算重采样信号
    resampled_signal = A.dot(resampled_signal)

    # 归一化处理
    resampled_signal /= np.max(resampled_signal)
    return resampled_signal
