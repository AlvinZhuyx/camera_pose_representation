#!/usr/bin/env bash
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
for ((i=0;i<${#scenes[@]};++i));
do
    scene=${scenes[i]}
    python train.py --dataset 7Scenes --scene $scene --config_file configs/posenet.ini --model posenet --device 0 --learn_beta --learn_gamma --train True
done
