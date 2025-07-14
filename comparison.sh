#!/bin/bash

source /home/zzh/anaconda3/etc/profile.d/conda.sh
conda activate SAHCD

sst_train=(
  --network 'sst'
  --dataset "Farmland"
  --flag_test "train"
  --epoches 400
  --patches 5
  --learning_rate 5e-4
  --gamma 0.9
  --weight_decay 0
)
sst_test=(
  --network 'sst'
  --dataset "Farmland"
  --flag_test "test"
  --epoches 400
  --patches 5
  --learning_rate 5e-4
  --gamma 0.9
  --weight_decay 0
)
#python main.py "${sst_train[@]}"
python main.py "${sst_test[@]}"