import pandas as pd
import os

picpath='C:/TOR/1290'
fpath='./input/urls_top_1m.csv'
log_path='./uncrawled.log'
df = pd.read_csv(fpath, nrows=1290,header=None)
old_list=df.values
f_list=[]
for f in old_list:
    f_list.append(f[0])

log=open(log_path,encoding="utf-8",mode="a")
pic_list = str(os.listdir(picpath))
pic_list = pic_list.replace(".png", "")

uncrawled=[i for i in f_list if not i in pic_list]
for un in uncrawled:
    log.write(un+'\n')
    print(un)
