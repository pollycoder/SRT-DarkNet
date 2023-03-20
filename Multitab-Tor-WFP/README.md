# Multitab-Tor-Website-Fingerprinting
Website fingerprinting attack against multi-tab tor browser 

## Environment configuration
**Note: disable all the automatic/background network traffic such as the auto-updates.**

### Download the Tor Browser
- You can download the latest Tor browser [here](https://www.torproject.org/download/).

- You can download any version of Tor Browser you want [here](https://archive.torproject.org/tor-package-archive/torbrowser/).

```shell
# Taking linux64 as an example, the installation package is: tor-browser-linux64-xx_en-US.tar.xz
mkdir tbb
mv tor-browser-linux64-xx_en-US.tar.xz tbb/
cd tbb && xz -d tor-browser-linux64-xx_en-US.tar.xz && tar -xvf tor-browser-linux64-xx_en-US.tar
chmod -R 777 tor-browser_en-US
```

### Download geckodriver
- You need to download the geckodriver that matches the tor browser version [here](https://archive.torproject.org/tor-package-archive/torbrowser/)

- Modify $PATH

```shell
vim ~/.bashrc
export PATH=$PATH:<path_to_geckodrive>
source ~/.bashrc
chmod 777 <path_to_geckodrive>
```

### Install required tools and packages

```shell
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install tcpdump wireshark xvfb ethtool tmux
pip install pyvirtualdisplay stem dpkt tld xvfbwrapper psutil selenium numpy
apt purge firefox && apt install firefox
```

### Packet capture settings

```shell
# set capture capabilities to your user
sudo chgrp root /usr/bin/dumpcap
sudo chmod 750 /usr/bin/dumpcap
sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/dumpcap
# change MTU to standard ethernet MTU (1500 bytes):
sudo ifconfig eth0 mtu 1500
# disable offloads
sudo ethtool -K eth0 tx off tso off gso off
```

## How to use
```shell
git clone git@github.com:Xinhao-Deng/Multitab-Tor-WFP.git
```

### Data collection
- configure
```python
# in helper/utils.py
MY_IP = "xxx.xx.x.xx"  # (run 'ifconfig' --> eth0 inet)
USED_GATEWAY_IP = "xxx.xx.xx.xxx" # (run 'route -n')

# in pipeline.sh
result_path='xxx' # path to save data
num_batch=1 # number of batches
num_tab=3 # number of tabs
num_comb=100 # number of site combinations
num_instance=20 # number of repeat visits
tbbpath='/tbb/tor-browser_en-US' # path to tor browser
```

- start data collection
```shell
cd ..
chmod -R 777 Multitab-Tor-WFP/
cd Multitab-Tor-WFP/data_collection
sh pipeline.sh
```

### Feature extraction
```shell
# extract features from .pcap
cd feature_extraction
python extractor.py config.json
# combine features
python combiner.py config.json
# K-Folds cross-validation
python split.py
```

### Attack
**DF [1]**
```shell
# train
python attack/df/train.py -g {device}
# test
python attack/df/test.py -g {device}
```

**DROPS**



## References
[1] Sirinam, Payap, et al. "Deep fingerprinting: Undermining website fingerprinting defenses with deep learning." Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security. 2018.