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
sys.path.append("/data/users/dengxinhao/research/early_wf_attack/attack")
import warnings

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
warnings.filterwarnings('ignore')

#* My code
import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.DF import DF
#* My code ends here

parser = argparse.ArgumentParser(description='Implementation of DF')
parser.add_argument("-g", '--gpu', default=0, type=int, help='Device id')
parser.add_argument("-d", '--dataset', default="DF", type=str, help='dataset name')
parser.add_argument("-l", '--length', default=5000, type=int, help='length of features')
parser.add_argument("-t", '--type', default="df", type=str, help='df of tiktok')
parser.add_argument("-e", '--enhance', default="F", type=str, help='enhance or not')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
length_features = args.length
method_type = args.type
is_enhance = args.enhance
assert method_type in ["df", "tiktok"]
assert is_enhance in ["T", "F"]

dataset_name = args.dataset
batch_size = 256
learning_rate = 0.002
num_epoch = 30
#! Senior fellow's code
'''
in_path = f"/data/users/dengxinhao/research/early_wf_attack/datasets/processed/{dataset_name}/pre_attack" # 实验packet
if is_enhance == "F":
    train_path = os.path.join(in_path, f"train.npz")
else:
    train_path = os.path.join(in_path, f"enhanced_train.npz")
valid_path = os.path.join(in_path, f"valid.npz")
networks_path = f'/data/users/dengxinhao/research/early_wf_attack/datasets/processed/{dataset_name}/networks'
os.makedirs(networks_path, exist_ok=True)
'''
#! Senior fellow's code ends here

def load_array(features, labels, batch_size, is_train = True):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train)

def measurement(y_true, y_pred):
    result = {
        "accuracy": round(accuracy_score(y_true, y_pred)*100, 3), 
        "precision": round(precision_score(y_true, y_pred, average="weighted")*100, 3), 
        "recall": round(recall_score(y_true, y_pred, average="weighted")*100, 3),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted")*100, 3)
    }
    return result

# 加载数据
#! Senior fellow's code
'''
train_data = np.load(train_path)
valid_data = np.load(valid_path)
train_X = train_data["X"][:,:length_features]
train_y = train_data["y"]
valid_X = valid_data["X"][:,:length_features]
valid_y = valid_data["y"]
'''
#! Senior fellow's code ends here
#* My code
dataset_name = "Front"
train_X, train_y, test_X, test_y, valid_X, valid_y = dataset(db_name=dataset_name, filter='butter-low')
#* My code ends here

if method_type == "df":
    train_X[train_X>0] = 1
    train_X[train_X<0] = -1
    valid_X[valid_X>0] = 1
    valid_X[valid_X<0] = -1

print(f"train: X:{train_X.shape}, y:{train_y.shape}")
print(f"valid: X:{valid_X.shape}, y:{valid_y.shape}")

assert train_y.max() == valid_y.max()
num_classes = train_y.max() + 1

# 数据转换
train_X = torch.tensor(train_X[:,np.newaxis], dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.int64)
valid_X = torch.tensor(valid_X[:,np.newaxis], dtype=torch.float32)
valid_y = torch.tensor(valid_y, dtype=torch.int64)

train_iter = load_array(train_X, F.one_hot(train_y, num_classes).type(torch.float32), batch_size)
train_valid_iter = load_array(train_X, F.one_hot(train_y, num_classes).type(torch.float32), batch_size, False)
valid_iter = load_array(valid_X, F.one_hot(valid_y, num_classes).type(torch.float32), batch_size, False)

# 读取模型并训练
model = DF(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.cuda()

for epoch in range(num_epoch):
    model.train()
    sum_loss = 0
    sum_count = 0
    for index, cur_data in enumerate(train_iter):
        cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
        optimizer.zero_grad()
        outs = model(cur_X)
        loss = criterion(outs, cur_y)
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]
    train_loss = round(sum_loss/sum_count, 3)
    
    # 验证当前epoch在train和valid的结果
    with torch.no_grad():
        model.eval()
        preds_train = []
        for index, cur_data in enumerate(train_valid_iter):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            outs = model(cur_X)
            outs = torch.argsort(outs, dim=1, descending=True)[:,0]
            preds_train.append(outs.cpu().numpy())
        preds_train = np.concatenate(preds_train).flatten()
        
        sum_loss = .0
        sum_count = 0
        preds_valid = []
        for index, cur_data in enumerate(valid_iter):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            outs = model(cur_X)
            loss = criterion(outs, cur_y)
            outs = torch.argsort(outs, dim=1, descending=True)[:,0]
            sum_loss += loss.data.cpu().numpy() * outs.shape[0]
            sum_count += outs.shape[0]
            preds_valid.append(outs.cpu().numpy())
        preds_valid = np.concatenate(preds_valid).flatten()
        test_loss = round(sum_loss/sum_count, 4)

        train_result = measurement(train_y.cpu().numpy(), preds_train)
        valid_result = measurement(valid_y.cpu().numpy(), preds_valid)
        print(f"epoch {epoch}: train_loss = {train_loss}, test_loss = {test_loss}")
        print("train_result:", train_result)
        print("valid_result:", valid_result)

#! Senior fellow's code
'''
torch.save(model.state_dict(), os.path.join(networks_path, f'{dataset_name}_{method_type}.pth'))
'''
#! Senior fellow's code ends here
torch.save(model.state_dict(), f=os.path.join("./", f'{dataset_name}_{method_type}.pth'))