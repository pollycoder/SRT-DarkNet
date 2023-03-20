import numpy as np 
import pandas as pd 

infile='dataset/merge/2021_open.npz'
data = np.load(infile, allow_pickle=True)
labels = data['labels']


print('labels:', labels.shape)
print('average:', labels.shape[0]/labels.shape[1])
for i in range(101):
    print(f'{i}: {labels[:,i].sum()}')
