from sklearn import manifold
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import random

######################
# Scattering module
# Scatter
# RGBs
######################

# Sample choosing and plotting
def sample_scatter(X_train, y_train, X_test, y_test, y_pred, X_train_raw, X_test_raw, n):
    print("Start plotting...")
    random_list = random.sample(range(1,np.max(y_train) + 1),n)
    print("Random web: ", random_list)
    X_plot_train = []
    y_plot_train = []
    X_plot_raw = []
    for i in trange(0, X_train.shape[0]):
        if y_train[i] in random_list:
            x_list = X_train[i,:].tolist()
            x_raw = X_train_raw[i,:].tolist()
            X_plot_train.append(x_list)
            y_plot_train.append(y_train[i])
            X_plot_raw.append(x_raw)
    X_plot_train = np.array(X_plot_train)
    y_plot_train = np.array(y_plot_train)
    X_plot_raw = np.array(X_plot_raw)
            

    X_plot_test = []
    y_plot_test = []
    y_plot_pred = []
    X_plot_rawtest = []
    for i in trange(0, X_test.shape[0]):
        if y_test[i] in random_list:
            x_list = X_test[i,:].tolist()
            x_rawtest = X_test_raw[i,:].tolist()
            X_plot_test.append(x_list)
            X_plot_rawtest.append(x_rawtest)
            y_plot_test.append(y_test[i])
            y_plot_pred.append(y_pred[i])
    X_plot_test = np.array(X_plot_test)
    y_plot_test = np.array(y_plot_test)
    X_plot_rawtest = np.array(X_plot_rawtest)
    return X_plot_train, y_plot_train, X_plot_test, y_plot_test, X_plot_raw, X_plot_rawtest


def showScatter(train_data, train_color, test_data, test_color, title, score, i, n, max):
    if train_data.shape[1] > 2:
        train_tsne = manifold.TSNE(n_components=2, 
                                   init='pca', 
                                   random_state=501).fit_transform(train_data)      # Dimensionality reduction of training data
    else:
        train_tsne = train_data
    if test_data.shape[1] > 2:
        test_tsne = manifold.TSNE(n_components=2, 
                                  init='pca', 
                                  random_state=501).fit_transform(test_data)        # Dimensionality reduction of testing data
    else:
        test_tsne = test_data

    plt.subplot(1, 2, i)
    finalTitle = title + "(n=" + str(n) + ")\nAccuracy = " + str(score)
    plt.title(finalTitle)
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_color, marker='o', label='training data', edgecolors='k')
    plt.legend()
    plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_color, marker='v', label='testing data', edgecolors='k')
    plt.legend()
    plt.colorbar()
    maxrange = max
    ti = np.arange(-maxrange, maxrange + 1, 20)
    plt.xlim(-maxrange, maxrange)  # 设定绘图范围
    plt.ylim(-maxrange, maxrange) 
    plt.xticks(ti)  # 设定刻度
    plt.yticks(ti)
    ax = plt.gca()
    ax.set_aspect(1)


# RGB for all pages
def rgb(X_train, y_train, min, max, width, title):
    j = 0
    rgb_matrix = []
    for i in trange(min, max + 1):
        rgb_array = np.zeros(X_train.shape[1])
        k = 0
        for j in trange(0, X_train.shape[0]):
            if y_train[j] == i:
                k += 1
                rgb_array = np.add(rgb_array, X_train[j, :])
        rgb_array = rgb_array / (k + 1)
        rgb_array = rgb_array.tolist()
        rgb_matrix.append(rgb_array)
    rgb_matrix = np.array(rgb_matrix)

    b = width
    y = np.arange(min, max + 1, 1)
    x = np.arange(1, b + 1, 1)
    z = rgb_matrix[:, 0:b]
    plt.pcolormesh(x,y,z, cmap='jet', vmax=50, vmin=0)
    plt.colorbar()
    plt.title(title)


# RGB for single page
def rgb_singlepage(X_train, y_train, index, width, title):
    rgb_matrix = []
    k = 0
    for i in trange(0, X_train.shape[0]):
        if y_train[i] == index:
            k += 1
            rgb_array = X_train[i,:].tolist()
            rgb_matrix.append(rgb_array)
        if k == 100:
            break
    rgb_matrix = np.array(rgb_matrix)

    b = width
    y = np.arange(0, k, 1)
    x = np.arange(1, b + 1, 1)
    z = rgb_matrix[:, 0:b]
    plt.pcolormesh(x,y,z, cmap='jet', vmax=100, vmin=0)
    plt.colorbar()
    plt.title(title)