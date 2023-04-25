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
import warnings
import torch.nn.functional as F
sys.path.append("/data/users/dengxinhao/research/early_wf_attack/attack")

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
warnings.filterwarnings("ignore")

import sys
sys.path.append("../")
from tools.DF import DF

def load_array(features, labels, batch_size, is_train = True):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train)

def measurement(y_true, y_pred):
    result = {
        "accuracy": round(accuracy_score(y_true, y_pred)*100, 2), 
        "precision": round(precision_score(y_true, y_pred, average="weighted")*100, 2), 
        "recall": round(recall_score(y_true, y_pred, average="weighted")*100, 2),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted")*100, 2)
    }
    return result


parser = argparse.ArgumentParser(description='Implementation of DF')
parser.add_argument("-g", '--gpu', default=0, type=int, help='Device id')
parser.add_argument("-d", '--dataset', default="DF", type=str, help='dataset name')
parser.add_argument("-l", '--length', default=5000, type=int, help='length of features')
parser.add_argument("-t", '--type', default="df", type=str, help='df of tiktok')
parser.add_argument("-s", '--spectrum', default="none", type=str, help='spectrum used for training, freq, ps or none')
args = parser.parse_args()
length_features = args.length
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
method_type = args.type
dataset_name = args.dataset
batch_size = 256
num_classes = 95


# 读取模型并测试
assert args.spectrum in ['none', 'freq', 'ps']
print("Loading the model ...")
model = DF(num_classes)
if args.spectrum == 'none':
    model.load_state_dict(torch.load(os.path.join("./", f'{dataset_name}_{method_type}_raw.pth'), map_location='cpu'))
elif args.spectrum == 'freq':
    model.load_state_dict(torch.load(os.path.join("./", f'{dataset_name}_{method_type}_freq.pth'), map_location='cpu'))
elif args.spectrum == 'ps':
    model.load_state_dict(torch.load(os.path.join("./", f'{dataset_name}_{method_type}_ps.pth'), map_location='cpu'))
else:
    print("TypeERROR: Wrong spectrum type !")

model = model.cuda()
model.eval()
print("Model loading succeeded !")

# Loading the dataset
print("Loading the data ...")
print("Dataset:", dataset_name)
data_dir = "../npz/"
data_X = 0
if args.spectrum == 'none':
    print("Spectrum: None")
    data_X = np.load(data_dir + dataset_name + "_X_raw.npz")
elif args.spectrum == 'freq':
    print("Spectrum: Frequency spectrum")
    data_X = np.load(data_dir + dataset_name + "_X_freq.npz")
elif args.spectrum == 'ps':
    print("Spectrum: Power spectrum with correlation")
    data_X = np.load(data_dir + dataset_name + "_X_ps.npz")
else:
    print("TypeERROR: Wrong spectrum type !")
data_y = np.load(data_dir + dataset_name + "_y.npz")
test_X = data_X["test"]
test_y = data_y["test"]
print(f"test: X:{test_X.shape}, y:{test_y.shape}")
    

# 数据转换
test_X = torch.tensor(test_X[:,np.newaxis], dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.int64)
test_iter = load_array(test_X, F.one_hot(test_y, num_classes).type(torch.float32), batch_size, False)

# Testing
print("Start testing ...")
with torch.no_grad():
    preds = []
    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
        outs = model(cur_X)
        outs = torch.argsort(outs, dim=1, descending=True)[:,0]
        preds.append(outs.cpu().numpy())
    preds = np.concatenate(preds).flatten()
      
result = measurement(test_y.cpu().numpy(), preds)
print(f"Test=", result)