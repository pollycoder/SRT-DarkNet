import pandas as pd
import os
pic_path="C:/TOR/pictures"
csv_path="./urls.csv"
fileMap = {}
size=0
def getFlist(path):
    for  parent, dirnames, filenames in os.walk(path):
        for files in filenames:
          #  print('parent is %s, filename is %s' % (parent, files))
            #print('the full name of the file is %s' % os.path.join(parent, files))
            size = os.path.getsize(os.path.join(parent, files))
            fileMap.setdefault(files, size)
            filelist = sorted(fileMap.items(), key=lambda d: d[1], reverse=True)
    return filelist

if __name__ == '__main__':
    file_names = getFlist(pic_path)
    print(file_names)
    urls=[]
    for f in file_names:
        f1=f[0]
        nf=f1.replace(".png","")
        urls.append(nf)
        
    fout=open(csv_path,encoding="utf-8",mode="w")
    for u in urls:
        fout.write(u+'\n')
    
