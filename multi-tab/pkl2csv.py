import pickle as pkl
import pandas as pd

with open(r'../datasets/y_train_NoDef.pkl', "rb") as f:
	object = pkl.load(f, encoding='latin1')
df = pd.DataFrame(object)
df.to_csv(r'../datasets/y_train_NoDef.csv')
