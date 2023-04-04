# Reconstruction of papers
# Paper title: Fine-Grained Webpage Fingerprinting Using OnlyPacket Length Information of Encrypted Traffic
# Pseudocode: Algorithm 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import sklearn.datasets as dt
from sklearn import manifold
from utility import dataset_loading_multitab
import datetime


###############################################
# Preprocess the original dataset
# Get U0 sequence
# Turn all the up-link packages into 0
###############################################
# Get U0 sequence
def getU0seq(original_seq):
    sum = 0
    for i in range(0, len(original_seq)):
        if original_seq[i] < 0:
            original_seq[i] = 0
        sum += original_seq[i]
        original_seq[i] = sum
    return original_seq


# Process all the traffics into U0 sequence
def getU0seq_dataset(original_dataset):
    start = datetime.datetime.now()
    print("=========Start getting U0 sequences=========")
    result_dataset = []
    for i in original_dataset:
        result = getU0seq(i)
        result_dataset.append(result)
    try:
        print("U0 sequence get successfully !")
    except:
        print("Error in U0 seq getting !")
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    return result_dataset


###############################################
# Extract features from U0 sequence
# Get block features (B)
# Blocks:[S, E, U]
# Result for each U0 sequence
###############################################
def feature_extractor_B(U0_seq):
    # Initialize the three dictionaries
    A_dict = {}
    S_dict = {}
    E_dict = {}
    
    # Fill in the elements according to U0 sequence
    for i in range(0, U0_seq.shape[0]):
        if i < U0_seq.shape[0] - 1 and U0_seq[i] == U0_seq[i + 1]:
            if U0_seq[i] in A_dict.keys():
                A_dict[U0_seq[i]] += 1
            else:
                A_dict[U0_seq[i]] = 1
                S_dict[U0_seq[i]] = i
        if U0_seq[i] in A_dict.keys() and i < U0_seq.shape[0] - 1 and U0_seq[i] != U0_seq[i + 1]:
            E_dict[U0_seq[i]] = i
        elif i == U0_seq.shape[0] - 1:
            E_dict[U0_seq[i]] = i

    # Delete elements whose cumulative result < 4
    key_list = list(A_dict.keys())
    for i in key_list:                                     
        if A_dict[i] < 4:
            del A_dict[i]
            del S_dict[i]
            del E_dict[i]
    
    # Get blocks
    B = []
    for i in A_dict.keys():
        B_list = [S_dict[i], E_dict[i], i]
        B += B_list
    return B


def feature_extractor_B_dataset(U0_seqs):
    start = datetime.datetime.now()
    print("============Start getting blocks============")
    result = []
    for i in U0_seqs:
        temp = feature_extractor_B(i)
        result.append(temp)
    result = np.array(result, dtype=object)
    try:
        print("Blocks getting successfully !")
    except:
        print("Error in getting blocks !")
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    print("==============================================")
    return result


###############################################
# Extract features from U0 sequence dataset
# Get sequence features (SF)
# Dataset: [
# [U01, U02,...]
# ]
###############################################
# Result: 0 for b, 1 for d
def feature_extractor_bd(U0_seqs):
    start = datetime.datetime.now()
    print("=============Start getting b and d=============")
    sum_b = 0
    sum_d = 0
    for i in range(0, len(U0_seqs)):
        block = feature_extractor_B(U0_seqs[i])
        sum_b += block[0]
        sum_d += block[-2]
    b = int(sum_b / len(U0_seqs))
    d = int(sum_d / len(U0_seqs))
    try:
        print("b=", b)
        print("d=", d)
    except:
        print("Fail to get b and d !")
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    return [b, d]


# Result: SF feature
def feature_extractor_SF_rare(U0_seq, b, d):
    result = U0_seq[b:d]
    return result


def feature_extractor_SF(U0_seqs, b, d):
    start = datetime.datetime.now()
    print("=============Start getting SF==============")
    result = []
    for i in U0_seqs:
        temp = feature_extractor_SF_rare(i, b, d)
        result.append(temp)
    result = np.array(result)
    try:
        print("Get SF successfully !")
    except:
        print("Error in getting SF !")
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    print("==============================================")
    return result


###############################################
# Extract features from *original sequence*
# Get statistical feature (ST)
###############################################
# Get rough statistical feature
def feature_extractor_ST_rare(original_seq):
    min = np.min(original_seq)
    max = np.max(original_seq)
    mean = np.mean(original_seq)
    mad = stats.median_abs_deviation(original_seq)
    std = np.std(original_seq)
    var = np.var(original_seq)
    skew = stats.skew(original_seq)
    kurt = stats.kurtosis(original_seq)
    per = np.percentile(original_seq, q=50)
    result = [min, max, mean, mad, std, var, skew, kurt, per]
    return result


# Random forest, reserve the features contributing the most
# Get the index of features
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

    def select_feature(self, dataset):
        importance = self.importance
        threshold = 0.04
        dataset = np.array(dataset)
        X_select_index = np.where(importance > threshold)
        X_select = dataset[:, importance > threshold]
        dataset = np.array(dataset)
        return importance, X_select, threshold

# Original_seqs: [original_seq[0], original_seq[1], ......]
def feature_extractor_ST(original_seqs, webpages):
    start = datetime.datetime.now()
    print("===============Start getting ST==============")
    final_seqs = []
    for seq in original_seqs:
        temp = feature_extractor_ST_rare(seq)
        final_seqs.append(temp)
    print("Loading finished !")
    rf = randomForest()
    rf.build(final_seqs, webpages)
    importance, st_data = rf.select_feature(final_seqs)
    try:
        print("Get ST successfully !")
    except:
        print("Error in getting ST !")
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    print("==============================================")
    return importance, st_data


###############################################
# Final preprocessing upon the dataset
# B, SF, ST
# Result: [
# [B1, SF1, ST1],
# [B2, SF2, ST2],
# ......
# ]
###############################################
def preprocess_B_SF(X_data):
    # U0 dataset and blocks
    U0_dataset = getU0seq_dataset(X_data)                                      
    block_dataset = feature_extractor_B_dataset(U0_dataset)                     

    # SF Feature
    dataset_bd = feature_extractor_bd(U0_dataset)                               
    sf_dataset = feature_extractor_SF(U0_dataset, dataset_bd[0], dataset_bd[1])
    return block_dataset, sf_dataset

def dataset_preprocess(X_data, webpages, X_test):
    start = datetime.datetime.now()
    print("Start data processing !<<<<<<<<<<<<<<<<<<<")

    ## Dataset preprocessing
    block_dataset, sf_dataset = preprocess_B_SF(X_data)

    # ST Feature
    st_index, st_dataset = feature_extractor_ST(X_data, webpages)

    # Join all the features together
    print("Start joint the feature vector of dataset===========")
    block = block_dataset[0]
    sf = sf_dataset[0]
    st = st_dataset[0]
    feature_vector = np.concatenate((sf, st))
    final_dataset = feature_vector
    for i in range(1, len(X_data)):
        block = block_dataset[i]
        sf = sf_dataset[i]
        st = st_dataset[i]
        feature_vector = np.concatenate((sf, st))
        final_dataset = np.row_stack((final_dataset, feature_vector))
        if i % 4000:
            print("Epoch", i / 4000)
    print("-----------------------------------------")

    ## Preprocess testing data
    print("Start to join the feature vector of testset===========")
    threshold = 0.04
    block_test, sf_test = preprocess_B_SF(X_test)
    final_seqs = []
    for seq in X_test:
        temp = feature_extractor_ST_rare(seq)
        final_seqs.append(temp)
    final_seqs = np.array(final_seqs)
    st_test = final_seqs[:, st_index > threshold]
    print("Test shape:", st_test.shape)
    block = block_test[0]
    sf = sf_test[0]
    st = st_test[0]
    feature_vector = np.concatenate((sf, st))
    final_test = feature_vector
    for i in range(1, len(X_test)):
        block = block_test[i]
        sf = sf_test[i]
        feature_vector = np.concatenate((sf, st))
        final_test = np.row_stack((final_test, feature_vector))
    
    # Robust ensurance
    try:
        print("Data processed successfully !")
    except:
        print("Error!")

    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    return final_dataset, final_test







# Test module
if __name__ == '__main__':
    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = dataset_loading_multitab()
    dataset, testset = dataset_preprocess(X_train, y_train, X_test)
    print("Dataset print finished ! >>>>>>>>>>>>>>>>>>>>>>>>>")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(dataset, y_train)
    result = model.score(testset, y_test)
    print("Accuracy = ", result)
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    