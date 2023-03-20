import pandas as pd

path_old='./uncrawled.log'
path_new='./tobecrawled.log'
fin=open(path_old,encoding="utf-8",mode="r")
fout=open(path_new,encoding="utf-8",mode="w")
crawl_list=[]
old_list=fin.readlines()
crawl_list=list(set(old_list))
for c in crawl_list:
    fout.write(c)
