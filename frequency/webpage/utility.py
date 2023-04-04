import pickle as pkl
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def dataset_loading_nodef():
    print("Loading datasets ....")
    dataset_dir = "../../datasets/"
    with open(dataset_dir + "X_train_NoDef.pkl", 'rb') as handle:
        X_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "y_train_NoDef.pkl", 'rb') as handle:
        y_train = np.array(pkl.load(handle, encoding='latin1'))
    with open(dataset_dir + "X_test_NoDef.pkl", 'rb') as handle:
        X_test = np.array(pkl.load(handle, encoding='latin1'))
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
    

   

    

