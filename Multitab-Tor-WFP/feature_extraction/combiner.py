#%%
import os
import sys
import dpkt
import json
import socket
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import time
from multiprocessing import Process
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

#%%
inpath = 'dataset/single'
outpath = 'dataset/merge'
name = '2021_open'

logging.info(f'Data combiner: load data from {inpath}')
os.makedirs(outpath, exist_ok=True)
outfile = os.path.join(outpath, f'{name}.npz')
config_path = sys.argv[1]
with open(config_path) as fin:
    config = json.load(fin)
feat_size = config["feat_size"]
label_size = config['label_size']

features = np.zeros((0, feat_size))
labels = np.zeros((0, label_size))

for infile in tqdm(os.listdir(inpath)):
    if infile.split('_')[0] != "open":
        date = infile.split('_')[0]
        if date >= "202201":
            continue
    cur_data = np.load(os.path.join(inpath, infile), allow_pickle=True)
    cur_features = cur_data['features']
    cur_labels = cur_data['labels']
    features = np.append(features, cur_features, axis=0)
    labels = np.append(labels, cur_labels, axis=0)

print(features.shape, labels.shape)
np.savez(outfile, features=features, labels=labels)