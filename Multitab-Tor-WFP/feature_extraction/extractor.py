#%%
import os
import sys
import dpkt
import json
import socket
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import time
from multiprocessing import Process
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

#%%
config_path = sys.argv[1]
logging.info(f"Data processor: load config from {config_path}")
with open(config_path) as fin:
    config = json.load(fin)

date = config["date"]
ntab = config["ntab"]
scenario = config["scenario"]
inpath = config["inpath"]
outpath = config["outpath"]
feat_size = config["feat_size"]
label_size = config['label_size']
min_feat_size = config["min_feat_size"]
client_version = config["client_version"]
client_ip = config["client_ip"][client_version]
cell_len = config["cell_len"]
cell_dir = config["cell_dir"]
closed_world_file = config["closed_world_file"]

df = pd.read_csv(closed_world_file, header=None)
df.columns = ['website']
website_id = {}
for idx, cur_web in enumerate(df['website']):
    website_id[cur_web] = idx

os.makedirs(outpath, exist_ok=True)
__decoder = {dpkt.pcap.DLT_LOOP: dpkt.loopback.Loopback,
             dpkt.pcap.DLT_NULL: dpkt.loopback.Loopback,
             dpkt.pcap.DLT_EN10MB: dpkt.ethernet.Ethernet,
             dpkt.pcap.DLT_LINUX_SLL: dpkt.sll.SLL}

def get_feature(pcap_file):
    res_sequence = {}
    seen_seq = set()
    
    st_time = None
    try:
        with open(pcap_file, 'rb') as fin:
            pcapin = dpkt.pcap.Reader(fin)
            decode = __decoder[pcapin.datalink()]
            for ts, buf in pcapin:
                if st_time is None:
                    st_time = ts - 1
                
                pkt = decode(buf)
                ip = pkt.data
                if not ip.p == dpkt.ip.IP_PROTO_TCP:
                    continue
                tcp = ip.data
                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue
                record_length = len(tcp.data)
                if tcp.seq in seen_seq:
                    continue
                if record_length == 0:
                    continue
                seen_seq.add(tcp.seq)
                
                now_dir = 'IN'
                web_ip = socket.inet_ntoa(ip.src)
                if socket.inet_ntoa(ip.src) in client_ip:
                    now_dir = 'OUT'
                    web_ip = socket.inet_ntoa(ip.dst)
                
                if not web_ip in res_sequence:
                    res_sequence[web_ip] = []

                res_sequence[web_ip].append(cell_dir[now_dir] * (ts-st_time))
    except KeyboardInterrupt:
        logging.info("WARNING\tKeyboard interrupt! Quitting in {}...".format(utils.cal_now_time()))
    except Exception as e:
        return np.zeros(0), 0

    final_sequence = []
    for key in res_sequence:
        if len(res_sequence[key]) > len(final_sequence):
            final_sequence = res_sequence[key]
    
    if len(final_sequence) < min_feat_size:
        return np.zeros(0), 0
    
    feature = np.zeros((1, feat_size))
    final_sequence = np.array(final_sequence)
    res_length = min(feat_size, len(final_sequence))
    feature[0][0:res_length] = final_sequence[0:res_length]
    return feature, len(final_sequence)

def process(args):
    paths = [path_list[i] for i in range(*args)]
    for cur_path in paths:
        length_list = []
        name = ('_').join(cur_path.split('/')[-2:])
        outfile = os.path.join(outpath, name + ".npz")
        features = np.zeros((0, feat_size))
        labels = np.empty((0, label_size))
        for cur_dir, dirs, files in os.walk(cur_path):
            if cur_dir.split("/")[-1].split("-")[0] == "comb":
                logging.info(f'processing...{cur_dir}')
                label_file = os.path.join(cur_dir, "label")
                assert os.path.isfile(label_file)
                with open(label_file, "r") as fp:
                    lines = fp.readlines()
                    cur_webs = lines[0].strip().split(' ')
                cur_label = np.zeros((1, label_size))
                for cur_web in cur_webs:
                    if cur_web in website_id:
                        cur_idx = website_id[cur_web]
                    else:
                        cur_idx = 100
                    cur_label[0][cur_idx] = 1
                for son_dir in dirs:
                    assert son_dir.split('-')[0] == 'inst'
                    pcap_file = os.path.join(cur_dir, son_dir, 'tcp.pcap')

                    check_flag = False
                    for screenshot_id in range(ntab):
                        cur_screenshot = os.path.join(cur_dir, son_dir, str(screenshot_id) + '.png')
                        if os.path.exists(cur_screenshot) is False:
                            check_flag = True
                            break
                    if check_flag:
                        continue
                    cur_feature, cur_length = get_feature(pcap_file)
                    if cur_feature.shape[0] == 0: # 不满足条件则把第一维置0
                        continue
                    features = np.append(features, cur_feature, axis=0)
                    labels = np.append(labels, cur_label, axis=0)
                    length_list.append(cur_length)
        logging.info(f'{cur_path}: features: {features.shape}, labels: {labels.shape}')
        np.savez(outfile, features=features, labels=labels)
        length_list = pd.Series(length_list)
        logging.info(length_list.describe())
# %%
path_list = []
if scenario == "close":
    cur_path = os.path.join(inpath, f"{date}_{ntab}tab")
elif scenario == "open":
    cur_path = os.path.join(inpath, f"{scenario}_{date}_{ntab}tab")
for cur_dir in os.listdir(cur_path):
    d = os.path.join(cur_path, cur_dir)
    assert os.path.isdir(d)
    path_list.append(d)


m = config.get("pool", 20)
n = len(path_list)
if n < m:
    m = n
n_section, extras = np.divmod(n, m)
splits = np.array([0] + extras * [n_section + 1] + (m - extras) * [n_section]).cumsum()
assert len(splits) <= m + 1
cur_ranges = list(zip(splits, splits[1:]))

pool: List[process] = []
for cur_range in cur_ranges:
    p = Process(target=process, args=(cur_range,))
    pool.append(p)
    p.start()

while True:
    children_alive = [p.is_alive() for p in pool]
    if not any(children_alive):
        logging.info("all child process done, exit!")
        break

    for i, alive in enumerate(children_alive):
        if not alive:
            logging.warning(f"process {i} not alive, args: {cur_ranges[i]}")

    logging.info("check child process alive done! sleep 60 seconds")
    time.sleep(60)
