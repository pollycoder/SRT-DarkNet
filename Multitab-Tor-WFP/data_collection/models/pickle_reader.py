import pickle
import pandas as pd




# x1,x2,x3,x4,x5=pickle.load(f1)
# y1,y2,y3,y4,y5=pickle.load(f2)

f1=pd.read_pickle('../input/webpage_2-5tab.pickle')
f1=pd.DataFrame(f1)
f1.to_csv("../input/webpage_2-5tab.csv")

#1w个1tab,5000个2tab，3333ge 3tab,2500个4tab,2000个5tab,共22833组合
#每个组合先搜集10个instance？