#!/usr/bin/env bash
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
for ((i=0;i<${#scenes[@]};++i));
do
    scene=${scenes[i]}
    ckpt="./logs_and_checkpoints_tosubmit1/7Scenes_${scenes[i]}_posenet_posenet_learn_beta_learn_gamma/epoch_060.pth.tar"
    python train.py --dataset 7Scenes --scene $scene --config_file configs/posenet.ini --model posenet --device 0 --learn_beta --learn_gamma --checkpoint $ckpt
done
