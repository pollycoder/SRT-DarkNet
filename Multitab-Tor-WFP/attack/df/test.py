import os
import sys
import math
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from model import DF
from sklearn.metrics import roc_auc_score
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

length_for_df = {
    5000:18,
    10000:37,
    15000:57,
    20000:76,
}

data_file='dataset/merge/2021_close.npz'
index_file='dataset/index/0_2021_close.npz'
num_tab = 5
networks_path = 'networks'

def load_array(features, labels, batch_size, is_train = True):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train)

parser = argparse.ArgumentParser(description='Implementation of DF')
parser.add_argument("-g", '--gpu', default=0, type=int, help='Device id')
parser.add_argument("-l", '--length', default=5000, type=int, help='length of features')
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
length_features = args.length
batch_size = 256

logging.info("start...loading features and labels")
data = np.load(data_file, allow_pickle=True)
data_index = np.load(index_file, allow_pickle=True)
features = data["features"][:,:length_features]
labels = data["labels"]
test_index = data_index["test_index"]
X = features[test_index]
y = labels[test_index]
logging.info("finish...loading features and labels")

X = torch.tensor(X.reshape(-1,1,length_features), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

test_iter = load_array(X, y, batch_size, is_train=False)
num_classes = y.shape[1]

result_path = 'results'
os.makedirs(result_path, exist_ok=True)
net_file=os.path.join(networks_path, f'0_2021_close_{length_features}_df.pth')
assert os.path.exists(net_file)

net = DF(num_classes, length_for_df[length_features])
net.load_state_dict(torch.load(net_file, map_location='cpu'))
net.to(device)
net.eval()

max_k = 5
tp = [.0] * (max_k + 1)
ndcg = [.0] * (max_k + 1)
sum_count = 0
pred = np.zeros((0, num_classes))

with torch.no_grad():
    for index, test_data in enumerate(test_iter):
        features, labels = test_data[0].to(device), test_data[1].to(device)
        outs = net(features)
        
        pred = np.append(pred, outs.cpu().data, axis=0)
        
        sum_count += outs.shape[0]
        bias = torch.rand(labels.shape[0], max_k).to(device)
        new_labels = torch.cat((labels, bias), 1)
        
        for k in range(1, max_k +1):
            arg_outs = torch.argsort(outs, dim=1, descending=True)[:,:k]
            arg_labels = torch.argsort(new_labels, dim=1, descending=True)[:,:max_k]

            for outs_index in range(k):
                for labels_index in range(max_k):
                    tp[k] += torch.sum(arg_outs[:,outs_index].data == arg_labels[:,labels_index].data)
                    ndcg[k] += (1.0/math.log(outs_index+2,2)) * torch.sum(arg_outs[:,outs_index].data == arg_labels[:,labels_index].data)


fp = open(os.path.join(result_path, f'multi_{num_tab}tab_{length_features}'), 'w')
MAP = .0
y_true = y.numpy()
print('AUC: {}'.format(roc_auc_score(y_true, pred, average='micro')))
fp.write('AUC: {}\n'.format(roc_auc_score(y_true, pred, average='micro')))

for k in range(1, max_k+1):
    Pk = tp[k]/ (sum_count * k)
    NDCGk = ndcg[k] / (sum_count * k)
    MAP += Pk
    print("P@{}: {}, NDCG@{}:{}, MAP@{}: {}".format(k, Pk, k, NDCGk, k, MAP/k))
    fp.write("P@{}: {}, NDCG@{}:{}, MAP@{}: {}\n".format(k, Pk, k, NDCGk, k, MAP/k))
fp.close()