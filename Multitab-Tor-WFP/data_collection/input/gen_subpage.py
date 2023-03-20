import pandas as pd
import json

#infile = "/data/users/dengxinhao/research/Multitab-Tor-WFP/data_collection/input/urls-top-100.csv"
infile = "/root/Multitab-Tor-WFP/data_collection/input/subpages.json"

with open(infile, 'r') as fp:
    subpages = json.load(fp)

for web in subpages:
    print(web)
