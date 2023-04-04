import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as dt
from sklearn import manifold


class randomForest():
    def __init__(self) -> None:
        pass

    def build(self, dataset, label):
        self.model = RandomForestClassifier(n_estimators=10000, 
                                            random_state=0, 
                                            n_jobs=-1)
        self.model.fit(dataset, label)
        self.importance = self.model.feature_importances_
        return self.model

    def importance(self):
        return self.importance

    def select_feature(self, dataset):
        importance = self.importance
        self.threshold = 0.04
        X_select = dataset[:, importance > self.threshold]
        return X_select


# Test the classifier
def test_rf():
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

    # Testing the importances
    rf = randomForest()
    rfmodel = rf.build(train_data, train_label)
    importance = rf.importance
    rank = rf.select_feature(train_data)
    print(importance)
    print(rank)

if __name__ == '__main__':
    test_rf()