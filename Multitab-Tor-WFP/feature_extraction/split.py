import os
import pickle
import logging
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

infile='dataset/merge/2021_close.npz'
outpath='dataset/final'
k=10
random_state=8
data = np.load(infile, allow_pickle=True)
X = data["features"]
y = data["labels"]

skf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
idx = 0
for train_index, test_index in skf.split(X, y):
    print(f'shape: train:{train_index.shape}, test:{test_index.shape}')
    cur_dict = {
        "train_X": X[train_index],
        "train_y": y[train_index],
        "test_X" : X[test_index],
        "test_y" : y[test_index]
    }

    outfile = os.path.join(outpath, str(idx) + '_' + infile.split('/')[-1].split('.')[0] + '.pkl')
    with open(outfile, "wb") as fp:
        pickle.dump(cur_dict, fp)
    #np.savez(outfile, train_index=train_index, test_index=test_index)
    idx += 1
    del cur_dict