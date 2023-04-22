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

cd ../DF
nohup python DF_exp.py > ../result/log/DF_mlp.log 2>&1 &

cd ../WTF_PAD
nohup python WTF_PAD_exp.py > ../result/log/WTF_PAD_mlp.log 2>&1 &

cd ../Front
nohup python Front_exp.py > ../result/log/Front_mlp.log 2>&1 &