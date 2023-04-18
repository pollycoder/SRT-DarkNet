#########################################
# Script for running the whole experiment
# Nodef		-	Raw
# Nodef 	- 	PSD
# WTF-PAD	-	Raw
# WTF-PAD	-	PSD
#########################################
rm -rf result
mkdir result
mkdir result/log
mkdir result/scatter
mkdir result/rgb
mkdir result/psd
cd DF
nohup python -u DF_exp.py > ../result/log/DF_exp.log 2>&1 &
#python -u DF_raw_exp.py > ../result/log/DF_raw_exp.log 2>&1 &
cd ../WTF_PAD
#python -u WTF_PAD_raw_exp.py > ../result/log/WTF_PAD_raw_exp.log 2>&1 &
nohup python -u WTF_PAD_exp.py > ../result/log/WTF_PAD_exp.log 2>&1 &
cd ../Front
#python -u Front_raw_exp.py > ../result/log/Front_raw_exp.log 2>&1 &
nohup python -u Front_exp.py > ../result/log/Front_exp.log 2>&1 &
