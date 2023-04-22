from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime

'''
APIs for differnet classifiers
    K-NN
    Random forest
    DNN - MLP
    Linear regression
'''


# KNN
def kNN(X_train, y_train, X_test, y_test):
    print("Start training (kNN)")
    model = KNeighborsClassifier(n_neighbors=5)
    start = datetime.datetime.now()
    model.fit(X_train, y_train)
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")
    print("Now start testing...")
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('Tesing time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc


def RF(X_train, y_train, X_test, y_test):
    print("Start training (RF)")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
    start = datetime.datetime.now()
    model.fit(X_train, y_train)
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")
    print("Now start testing...")
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('Tesing time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc



def DNN(X_train, y_train, X_test, y_test):
    print("Start training (MLP)")
    
    start = datetime.datetime.now()
    mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(200,200), early_stopping=True, random_state=420)
    mlp.fit(X_train, y_train)
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")

    print("Now start testing...")
    start = datetime.datetime.now()
    acc = mlp.score(X_test, y_test)
    print("Acc={:.2f}".format(acc))
    y_pred = mlp.predict(X_test)
    end = datetime.datetime.now()
    print('Testing time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc


def DT(X_train, y_train, X_test, y_test):
    print("Start training (DT)")
    model = tree.DecisionTreeClassifier()
    start = datetime.datetime.now()
    model.fit(X_train, y_train)
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")
    print("Now start testing...")
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('Testing time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc


def LR(X_train, y_train, X_test, y_test):
    print("Start training (LR)")
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    start = datetime.datetime.now()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")
    print("Now start testing...")
    start = datetime.datetime.now()
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    acc = model.score(X_test_scaled, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('Testing time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc
