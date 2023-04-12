from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import datetime


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
    # 定义MLP分类器
    mlp = MLPClassifier(max_iter=1000)

    # 定义要搜索的参数范围
    param_grid = {
    'hidden_layer_sizes': [(100, 100),(150, 150), (200, 200), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    }

    # 使用网格搜索来搜索最佳参数组合
    grid_search = GridSearchCV(mlp, param_grid=param_grid, cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和训练集上的准确率
    print("Best parameter: ", grid_search.best_params_)
    print("Train accuracy: {:.2f}".format(grid_search.best_score_))
    end = datetime.datetime.now()
    print("Training succeeded !")
    print('Training time: ', (end - start).seconds, "s")

    # 在测试集上评估性能
    print("Now start testing...")
    start = datetime.datetime.now()
    acc = grid_search.score(X_test, y_test)
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