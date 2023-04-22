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
python DF_train.py
nohup python -u DF_test.py > ../result/log/DF.log 2>&1 &

cd ../WTF_PAD
python WTF_PAD_train.py
nohup python -u WTF_PAD_test.py > ../result/log/WTF_PAD.log 2>&1 &

cd ../Front
python Front_train.py
nohup python -u Front_test.py > ../result/log/Front.log 2>&1 &
