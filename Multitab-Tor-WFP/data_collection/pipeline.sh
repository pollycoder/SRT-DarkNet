#!/bin/bash

cur_date="`date +%m%d%h`"
time=$(date)
echo $time"-----开始数据收集------" >> /root/crontab.log

#*********在这改动***********
start_comb=10000
end_comb=10249
server_name=XML12
#crawl从第几行到第几行 便于服务器协调
#*********在这改动***********


result_path='/root/tor_data/'${cur_date}_${server_name}

# num_batch=1
# num_comb=1
#batch和comb先删了
num_instance=10
#每个组合先搜集10个instance

tbbpath='/root/tbb/tor-browser_en-US'
url_path='./input/webpage_2-5tab.pickle'
#webpage是子网页组合 website是网页组合
#pickle文件



# remove data
rm -rf /root/tor_data/*

# Data collection
# conda activate py36

python data_collector.py --urls_check ${url_path} --output ${result_path} --tbbpath ${tbbpath}  --start_comb ${start_comb} --end_comb ${end_comb} --xvfb True  --instance ${num_instance} --screenshot True