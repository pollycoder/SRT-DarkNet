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
python data_pipeline.py -d WTF_PAD

# Training
cd ../DF_exp
python -u train.py -d WTF_PAD
python -u train.py -s freq -d WTF_PAD
python -u train.py -s ps -d WTF_PAD

# Testing
python -u test.py -d WTF_PAD
python -u test.py -s freq -d WTF_PAD
python -u test.py -s ps -d WTF_PAD


