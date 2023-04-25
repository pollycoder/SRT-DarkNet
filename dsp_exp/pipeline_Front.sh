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
python data_pipeline.py -d Front

# Training
cd ../DF_exp
python -u train.py -d Front
python -u train.py -s freq -d Front
python -u train.py -s ps -d Front

# Testing
python -u test.py -d Front
python -u test.py -s freq -d Front
python -u test.py -s ps -d Front


