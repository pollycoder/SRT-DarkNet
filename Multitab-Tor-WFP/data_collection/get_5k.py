import csv
import pandas as pd
import numpy as np
data=pd.read_csv('C:/TOR/Multitab-Tor-WFP/data_collection/input/urls_top_1m_old.csv',header=None)
outdir='C:/TOR/Multitab-Tor-WFP/data_collection/input/urls_top_2w.csv'
result =data.iloc[15000:20000,:]
res=result.values.tolist()
#print(res)
fout=open(outdir,encoding="utf-8",mode="w")
for c in res:
    if(c!=0):
        print(c)
        fout.write(c[0]+'\n')
