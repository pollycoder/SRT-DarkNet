from sklearn.neighbors import KNeighborsClassifier
import datetime

#####################################
# Classifiers
# KNN
#####################################

# KNN
def kNN(X_train, y_train, X_test, y_test):
    print("Start training (kNN)")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print("Training succeeded !")
    print("Now start testing...")
    start = datetime.datetime.now()
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print("Acc =", acc)
    end = datetime.datetime.now()
    print('KNN time: ', (end - start).seconds, "s")
    print("--------------------------------------")
    return y_pred, acc