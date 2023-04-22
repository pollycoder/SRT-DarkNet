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

#* My code
import sys
sys.path.append("../")
from tools.data_loading import dataset
from tools.DF import DF
#* My code ends here

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
args = parser.parse_args()
length_features = args.length
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
method_type = args.type
dataset_name = args.dataset
batch_size = 256
num_classes = 95

#! Senior fellow's code
'''
in_path = f"/data/users/dengxinhao/research/early_wf_attack/datasets/processed/{dataset_name}/in_attack"
networks_path = f'/data/users/dengxinhao/research/early_wf_attack/datasets/processed/{dataset_name}/networks'
'''
#! Senior fellow's code ends here

# 读取模型并测试
model = DF(num_classes)
#! Senior fellow's code
'''
model.load_state_dict(torch.load(os.path.join(networks_path, f'{dataset_name}_{method_type}.pth'), map_location='cpu'))
'''
#! Senior fellow's code ends here
#* My code
model.load_state_dict(torch.load(os.path.join("./", f'{dataset_name}_{method_type}.pth'), map_location='cpu'))
model = model.cuda()
model.eval()
#* My code ends here

'''
for percentile in [10, 30, 50, 70, 90, 100]:
    #! Senior fellow's code
    test_path = os.path.join(in_path, f"p{percentile}.npz")
    #! Senior fellow's code ends here

    # 加载数据
    #! Senior fellow's code
    test_data = np.load(test_path)
    test_X = test_data["X"][:,:length_features]
    test_y = test_data["y"]
    #! Senior fellow's code ends here
    #* My code
    dataset_name = "DF"
    train_X, train_y, test_X, test_y, valid_X, valid_y = dataset(db_name=dataset_name)
    print(f"test: X:{test_X.shape}, y:{test_y.shape}")
    #* My code ends here
    
    if method_type == "df":
        print("ok")
        test_X[test_X>0] = 1
        test_X[test_X<0] = -1

    # 数据转换
    test_X = torch.tensor(test_X[:,np.newaxis], dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.int64)
    test_iter = load_array(test_X, F.one_hot(test_y, num_classes).type(torch.float32), batch_size, False)


    with torch.no_grad():
        preds = []
        for index, cur_data in enumerate(test_iter):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            outs = model(cur_X)
            outs = torch.argsort(outs, dim=1, descending=True)[:,0]
            preds.append(outs.cpu().numpy())
        preds = np.concatenate(preds).flatten()
      
    result = measurement(test_y.cpu().numpy(), preds)
    print(f"result of {percentile}%:", result)
'''
#* My code
dataset_name = "DF"
train_X, train_y, test_X, test_y, valid_X, valid_y = dataset(db_name=dataset_name, filter='butter-low')
print(f"test: X:{test_X.shape}, y:{test_y.shape}")
#* My code ends here
    
#! Senior fellow's code
'''
if method_type == "df":
    print("ok")
    test_X[test_X>0] = 1
    test_X[test_X<0] = -1
'''
#! Senior fellow's code ends here

# 数据转换
test_X = torch.tensor(test_X[:,np.newaxis], dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.int64)
test_iter = load_array(test_X, F.one_hot(test_y, num_classes).type(torch.float32), batch_size, False)


with torch.no_grad():
    preds = []
    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
        outs = model(cur_X)
        outs = torch.argsort(outs, dim=1, descending=True)[:,0]
        preds.append(outs.cpu().numpy())
    preds = np.concatenate(preds).flatten()
      
result = measurement(test_y.cpu().numpy(), preds)
print(f"Accuracy=", result)