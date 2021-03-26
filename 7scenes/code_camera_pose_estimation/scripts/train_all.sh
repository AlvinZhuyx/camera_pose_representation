#!/usr/bin/env bash
scenes=("stairs" "fire" "heads" "office" "pumpkin" "redkitchen" "chess")
for ((i=0;i<${#scenes[@]};++i));
do
    scene=${scenes[i]}
    python train.py --dataset 7Scenes --scene $scene --config_file configs/posenet.ini --model posenet --device 0 --learn_beta --learn_gamma --train True
done
