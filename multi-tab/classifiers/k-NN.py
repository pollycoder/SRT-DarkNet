import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import manifold
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# Classifiers: K-NN
class kNN():
    def __init__(self, k) -> None:
        self.k = k

    def build(self, dataset, label):
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(dataset, label)
        return self.model


# Function visualizing the result
# 'o' for training data
# 'v' for testing data
def showScatter(train_data, train_color, test_data, test_color, title, score, i):
    plt.subplot(1, 2, i)
    finalTitle = title + "\nAccuracy = " + str(score)
    plt.title(finalTitle)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_color, marker='o')
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_color, marker='v')
    plt.legend()


# Test the classifier
def test_knn():
    '''
    dataset = dt.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.33, random_state=42)
    '''
    k = 10
    dataSet, label = dt.make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=[1.4, 1.5, 2.0])

	# Divide the dataset
    n_row, n_col = dataSet.shape
    train_data = dataSet[:-10]
    test_data = dataSet[n_row-10:]
    train_label = label[:-10]
    test_label = label[n_row-10:]

    # Preparation for plotting
    # If the dimension > 2, we need to make dimension reduction; 
    # else we can use the rare data directly
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

    # Testing
    score = kNN(k).build(train_data, 
                         train_label).score(test_data, test_label)
    y_predict = kNN(k).build(train_data, train_label).predict(test_data)
    
    # Plot
    showScatter(train_tsne, train_label, 
                test_tsne, test_label, 
                "True result for kNN", score, 1)  
    showScatter(train_tsne, train_label, 
                test_tsne, y_predict, 
                "Real result for kNN", score, 2)
    plt.show()


if __name__ == '__main__':
    test_knn()





