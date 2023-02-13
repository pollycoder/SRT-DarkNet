# Reconstruction of papers
# Paper title: Fine-Grained Webpage Fingerprinting Using OnlyPacket Length Information of Encrypted Traffic
# Pseudocode: Algorithm 1
import numpy as np
import matplotlib.pyplot as plt

##########################################
# Preprocess the original dataset
# Get U0 sequence
# Turn all the up-link packages into 0
##########################################
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
    result_dataset = []
    for i in original_dataset:
        result = getU0seq(i)
        result_dataset.append(result)
    return result_dataset


############################################
# Extract features from U0 sequence
# Get block features (B)
# Blocks:[S, E, U]
# Result for each U0 sequence
############################################
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
        B.append(B_list)
    return B


###############################################
# Extract features from U0 sequence dataset
# Get sequence features (SF)
# Dataset: [
# [U01, U02,...]
# ]
###############################################
# Result: 0 for b, 1 for d
def feature_extractor_bd(U0_seqs):
    sum_b = 0
    sum_d = 0
    for i in range(0, len(U0_seqs)):
        block = feature_extractor_B(U0_seqs[i])
        sum_b += block[0][0]
        sum_d += block[-1][1]
    b = int(sum_b / len(U0_seqs))
    d = int(sum_d / len(U0_seqs))
    return [b, d]


# Result: SF feature
def feature_extractor_SF(U0_seq, b, d):
    result = U0_seq[b:d]
    return result


################################################
# Extract features from *original sequence*
# Get statistical feature (ST)
################################################








# Test module
if __name__ == '__main__':
    array = np.array([[-66, 66, -54, -71, -60, -14, 14, -71, -72, -88, -65, 43, 45, 46, -54, -43],
                     [-66, -66, -54, -71, -60, -14, 14, -71, -72, -88, -65, 43, 45, 46, -54, -43],
                     [-66, 66, -54, -71, -60, -14, -14, -71, -72, -88, -65, 43, 45, 46, -54, -43],
                     [-66, 66, -54, -71, -60, -14, 14, -71, -72, -88, -65, -43, 45, 46, -54, -43]
                    ])
    dataset = getU0seq_dataset(array)
    bd_list = feature_extractor_bd(dataset)
    result = feature_extractor_SF(array[0], bd_list[0], bd_list[1])
    i = 0

    
    