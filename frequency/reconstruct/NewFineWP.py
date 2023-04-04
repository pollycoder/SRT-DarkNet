# Reconstruction of papers
# Paper title: Fine-Grained Webpage Fingerprinting Using OnlyPacket Length Information of Encrypted Traffic
# Pseudocode: Algorithm 1
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import sklearn.datasets as dt
from utility import dataset_loading_multitab
import datetime


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
        return importance, X_select


###############################################
# Preprocess the original dataset
# Get U0 sequence
# Turn all the up-link packages into 0
# Input: original sequences (np.array)
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


###############################################
# Extract features from *original sequence*
# Get statistical feature (ST)
# Input: original sequence (np.array)
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



###############################################
# Extract features from original sequences
# Input: original data (np.array)
###############################################
def data_preprocess(dataset, train_labels, testset):
    print("Start processing dataset>>>>>>>>>>>>>>>>>>>>>>>>>")
    U0_seqs = np.zeros(dataset.shape[1])            # U0 set
    blocks = []                                   # Blocks set
    st_seqs = []                                    # ST sequences set
    sum_b = 0                                       # Get b
    sum_d = 0                                       # Get d

    print("Start getting blocks and st lists")
    start = datetime.datetime.now()
    i = 0
    start_loop = datetime.datetime.now()
    for seq in dataset:    
        i += 1    
        # Get U0
        U0_seq = getU0seq(seq)
        block = feature_extractor_B(U0_seq)
        st_rare = feature_extractor_ST_rare(seq)

        # Process blocks and ST sequences
        U0_seqs = np.row_stack((U0_seqs, U0_seq))
        blocks.append(block)
        st_seqs.append(st_rare)

        # Cumulate b and d
        sum_b += block[0]
        sum_d += block[-2]
        st_rare = np.array(st_rare)
        
        if i % 1000 == 0:
            print("Epoch", i / 1000, end="  ")
            end_loop = datetime.datetime.now()
            print('Epoch time: ', (end_loop - start_loop).seconds, "s")
            start_loop = datetime.datetime.now()

    # Get b and d
    b = int(sum_b / len(dataset))
    d = int(sum_d / len(dataset))
    print("b=", b)
    print("d=", d)
    # Get SF features
    sf_seqs = U0_seqs[:, b:d]
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    print("Succeed\n")

    
    # Random forest ---> choose ST features
    print("Start selecting ST")
    start = datetime.datetime.now()
    rf = randomForest()
    rf.build(st_seqs, train_labels)
    importance, st_seqs, threshold = rf.select_feature(st_seqs)
    end = datetime.datetime.now()
    print('total time: ', (end - start) .seconds, "s")
    print("Succeed\n")

    # Join the features
    print("Start join the features")
    start = datetime.datetime.now()
    final_dataset = np.concatenate(sf_seqs[0], st_seqs[0])
    for i in range(1, len(dataset)):
        feature_vector = np.concatenate(sf_seqs[i], st_seqs[i])
        final_dataset = np.row_stack((final_dataset, feature_vector))
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    print("Succeed\n")
    print("Dataset finished !>>>>>>>>>>>>>>>>>>>>>>>>")
    print("==================================================")

    # Preprocess testset
    print("Start processing testset>>>>>>>>>>>>>>>>>>>>>>>>")
    start = datetime.datetime.now()
    U0_seqs_test = []
    blocks_test = []
    st_seqs_test = []
    for seq in testset:
        U0_seq_test = getU0seq(seq)
        block = feature_extractor_B(U0_seq_test)
        st_rare = feature_extractor_ST_rare(seq)

        U0_seqs_test.append(U0_seq_test)
        blocks_test.append(block)
        st_seqs_test.append(st_rare)
        

    st_seqs_test = st_seqs_test[:, importance > threshold]
    sf_seqs_test = U0_seqs_test[:, b:d]

    # Join the features
    final_testset = np.concatenate(sf_seqs_test[0], st_seqs_test[0])
    for i in range (1, len(testset)):
        feature_vector = np.concatenate(sf_seqs_test[i], st_seqs_test[i])
        final_testset = np.row_stack((final_testset, feature_vector))
    print('total time: ', (end - start).seconds, "s")
    print("Succeed\n")
    print("Testset finished>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return final_dataset, final_testset


# Test module
if __name__ == '__main__':
    start = datetime.datetime.now()
    X_train, y_train, X_test, y_test = dataset_loading_multitab()
    dataset, testset = data_preprocess(X_train, y_train, X_test)

    print("Start training")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(dataset, y_train)
    result = model.score(testset, y_test)
    print("Accuracy = ", result)
    end = datetime.datetime.now()
    print('total time: ', (end - start).seconds, "s")
    
    


