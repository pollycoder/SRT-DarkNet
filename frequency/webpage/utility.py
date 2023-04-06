import pickle as pkl
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import manifold
import matplotlib.pyplot as plt
from tqdm import trange


def dataset_loading_nodef():
    print("Loading datasets ....")
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_NoDef.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
        #X_train = X_train[:,1000:1100]
    with open(dataset_dir + "y_train_NoDef.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_NoDef.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
        #X_test = X_test[:,1000:1100]
    with open(dataset_dir + "y_test_NoDef.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


def dataset_loading_wtfpad():
    print("Loading datasets ....")
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_WTFPAD.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_WTFPAD.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_WTFPAD.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_test_WTFPAD.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


def dataset_loading_wt():
    print("Loading datasets ....")
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_WalkieTalkie.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_WalkieTalkie.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_WalkieTalkie.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_test_WalkieTalkie.pkl", 'rb') as handle:
        y_test = np.array(pkl.load(handle, encoding='latin1'))
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


def dataset_loading_multitab():
    print("Loading datasets...")
    dataset_dir = "../../datasets/1180filter/"
    with open(dataset_dir + "classifier.pickle", 'rb') as handle:
        train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "test.pickle", 'rb') as handle:
        test = np.array(pkl.load(handle, encoding='latin1')) 
    X_train = train[:, 1000:5000]
    y_train = train[:, 20000:20005]
    y_train = MultiLabelBinarizer().fit_transform(y_train)
    X_test = test[:, 1000:5000]
    y_test = test[:, 20000:20005]
    y_test = MultiLabelBinarizer().fit_transform(y_test)
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


def dataset_loading_newmultitab():
    print("Loading datasets...")
    dataset_dir = "../../datasets/all_data/"
    with open(dataset_dir + "classifier.pickle", 'rb') as handle:
        train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "test.pickle", 'rb') as handle:
        test = np.array(pkl.load(handle, encoding='latin1')) 
    X_train = train[:, 1000:5000]
    y_train = train[:, 20000:20005]
    y_train = MultiLabelBinarizer().fit_transform(y_train)
    X_test = test[:, 1000:5000]
    y_test = test[:, 20000:20005]
    y_test = MultiLabelBinarizer().fit_transform(y_test)
    print("Data loaded successfully !")
    return X_train, y_train, X_test, y_test


# Function visualizing the result
# 'o' for training data
# 'v' for testing data
# Preparation for plotting
# If the dimension > 2, we need to make dimension reduction; 
# else we can use the rare data directly
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