#########################################
# Script for running the whole experiment
# Nodef		-	Raw
# Nodef 	- 	PSD
# WTF-PAD	-	Raw
# WTF-PAD	-	PSD
#########################################
rm -rf result
mkdir result
cd nodef
nohup python -u nodef_exp.py > ../result/nodef_exp.log 2>&1 &
nohup python -u nodef_raw_exp.py > ../result/nodef_raw_exp.log 2>&1 &
cd ../wtfpad
nohup python -u wtfpad_raw_exp.py > ../result/wtfpad_raw_exp.log 2>&1 &
nohup python -u wtfpad_exp.py > ../result/wtfpad_exp.log 2>&1 &
cd ../wt
nohup python -u wt_raw_exp.py > ../result/wt_raw_exp.log 2>&1 &
nohup python -u wt_exp.py > ../result/wt_exp.log 2>&1 &
