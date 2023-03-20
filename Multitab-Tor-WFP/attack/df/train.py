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
os.makedirs(networks_path, exist_ok=True)


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
learning_rate = 0.002
num_epoch = 20

logging.info("start...loading features and labels")
data = np.load(data_file, allow_pickle=True)
data_index = np.load(index_file, allow_pickle=True)
features = data["features"][:,:length_features]
labels = data["labels"]
train_index = data_index["train_index"]
X = features[train_index]
y = labels[train_index]
logging.info("finish...loading features and labels")

X = torch.tensor(X.reshape(-1,1,length_features), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

train_iter = load_array(X, y, batch_size)
num_classes = y.shape[1]

net = DF(num_classes, length_for_df[length_features])
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adamax(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
net = net.to(device)
net.train()

for epoch in range(num_epoch):
    sum_loss = 0.0
    true_count = 0
    sum_count = 0
    for index, train_data in enumerate(train_iter):
        X, y = train_data[0].to(device), train_data[1].to(device)
        optimizer.zero_grad()
        outs = net(X)
        loss = criterion(outs, y)
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.data
        sum_count += outs.shape[0]
        
        bias = torch.rand(y.shape[0], num_tab).to(device)
        new_y = torch.cat((y, bias), 1)
        
        arg_outs = torch.argsort(outs, dim=1, descending=True)[:,:num_tab]
        arg_y = torch.argsort(new_y, dim=1, descending=True)[:,:num_tab]
        
        for outs_index in range(num_tab):
            for y_index in range(num_tab):
                true_count += torch.sum(arg_outs[:,outs_index].data == arg_y[:,y_index].data)
    logging.info('tp = {} sum = {}'.format(true_count, sum_count))
    logging.info('epoch %d: loss = %.03f P@K = %.03f' % (epoch, sum_loss / sum_count, 
                                                (true_count) / (sum_count * num_tab)))

name = index_file.split('/')[-1].split('.')[0]
torch.save(net.state_dict(), os.path.join(networks_path, f'{name}_{length_features}_df.pth'))
del net









