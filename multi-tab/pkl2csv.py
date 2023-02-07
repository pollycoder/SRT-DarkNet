import pickle as pkl
import pandas as pd

with open(r'./datasets/X_test_NoDef.pkl', "rb") as f:
	object = pkl.load(f)
df = pd.DataFrame(object)
df.to_csv(r'./datasets/X_test_NoDef.csv')
