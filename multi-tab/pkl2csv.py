import pickle as pkl
import pandas as pd
import numpy as np

with open(r'../datasets/1180filter/test.pickle', "rb") as f:
	object = pkl.load(f, encoding='latin1')
df = pd.DataFrame(object)
df.to_csv(r'../datasets/test.csv')

