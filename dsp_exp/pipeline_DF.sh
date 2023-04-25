#########################################
# Script for running the whole experiment
# DF
# WTF-PAD
# Front
#########################################
rm -rf result
mkdir result
mkdir result/log

cd tools
python data_pipeline.py

# Training
cd ../DF_exp
python -u train.py
python -u train.py -s freq
python -u train.py -s ps

# Testing
python -u test.py
python -u test.py -s freq
python -u test.py -s ps


