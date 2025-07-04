#!/bin/bash
# cd /home/xxxxxxxx/Uncertainty-aware-Blur-Prior

brain_backbone="EEGProjectLayer"
vision_backbone="RN50"

dataset="eeg"
i="01"
seed=0
python main.py --config configs/eeg/ubp.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/eeg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;

python main.py --config configs/eeg/ubp.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

python main.py --config configs/eeg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

dataset="meg"
i="01"
seed=0
python main.py --config configs/meg/ubp.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
python main.py --config configs/meg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;

python main.py --config configs/meg/ubp.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

python main.py --config configs/meg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

# brain_backbones=("EEGProjectLayer" "Shallownet" "Deepnet" "EEGnet" "TSconv")
# vision_backbones=("RN50" "RN101" "ViT-B-16" "ViT-B-32" "ViT-L-14" "ViT-H-14" "ViT-g-14" "ViT-bigG-14")
# for vision_backbone in "${vision_backbones[@]}"; do
#     for brain_backbone in "${brain_backbones[@]}"; do
#         for i in {01..10}; 
#         do 
#             for seed in {0..1}; 
#             do 
#                 echo "Running with brain_backbone=$brain_backbone, vision_backbone=$vision_backbone, subject=sub-$i, seed=$seed"
#                 python main.py --config baseline.yaml --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
#                 python main.py --config baseline.yaml --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

#                 python main.py --config ubp.yaml --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
#                 python main.py --config ubp.yaml --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;
#             done
#         done
#     done
# done

