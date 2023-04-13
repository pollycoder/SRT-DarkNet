#########################################
# Script for running the whole experiment
# Nodef		-	Raw
# Nodef 	- 	PSD
# WTF-PAD	-	Raw
# WTF-PAD	-	PSD
#########################################
rm -rf result
mkdir result
cd DF
nohup python -u DF_exp.py > ../result/DF_exp.log 2>&1 &
nohup python -u DF_raw_exp.py > ../result/DF_raw_exp.log 2>&1 &
cd ../WTF_PAD
nohup python -u WTF_PAD_raw_exp.py > ../result/WTF_PAD_raw_exp.log 2>&1 &
nohup python -u WTF_PAD_exp.py > ../result/WTF_PAD_exp.log 2>&1 &
cd ../Front
nohup python -u Front_raw_exp.py > ../result/Front_raw_exp.log 2>&1 &
nohup python -u Front_exp.py > ../result/Front_exp.log 2>&1 &
