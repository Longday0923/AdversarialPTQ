#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
# DATASET=cifar10
# NETWORK=AlexNet
# NETPATH=/gpfs/accounts/eecs598w23_class_root/eecs598w23_class/shared_data/zhtianyu/models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=1
# OPTIMIZ=Adam
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_channel_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(1.0)
# MARGINS=(5.0)
# DATANORM=True


# CIFAR10 - VGG16
# DATASET=cifar10
# NETWORK=VGG16
# NETPATH=/gpfs/accounts/eecs598w23_class_root/eecs598w23_class/shared_data/zhtianyu/models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=1
# OPTIMIZ=Adam
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_channel_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)
# DATANORM=True


# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18
# NETPATH=/gpfs/accounts/eecs598w23_class_root/eecs598w23_class/shared_data/zhtianyu/models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=1
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_channel_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)
# DATANORM=True


# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2
# NETPATH=/gpfs/accounts/eecs598w23_class_root/eecs598w23_class/shared_data/zhtianyu/models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=1
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_channel_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)
# DATANORM=True

# CIFAR10 - WideResNet
DATASET=cifar10
NETWORK=WideResNet
NETPATH=/gpfs/accounts/eecs598w23_class_root/eecs598w23_class/shared_data/zhtianyu/models/cifar10/train/model_cifar_wrn.pt
N_CLASS=10
BATCHSZ=128
N_EPOCH=1
OPTIMIZ=Adam
LEARNRT=0.00001
MOMENTS=0.9
O_STEPS=50
O_GAMMA=0.1
NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
W_QMODE='per_channel_symmetric'
A_QMODE='per_layer_asymmetric'
LRATIOS=(1.0)
MARGINS=(5.0)
DATANORM=false

# adversarial attack
att_type="PGD"
att_tar="untar"
att_step_size=0.05 # 0.05 in HW, 0.003 in TRADES
att_num_steps=10 # 10 in HW, 20 in TRADES
att_epsilon=0.031 # 0.3 in HW, 0.031 in TRADES


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
# for each_numrun in {1..2..1}; do       # it runs 10 times
each_numrun=1
for each_lratio in ${LRATIOS[@]}; do
for each_margin in ${MARGINS[@]}; do

  # : make-up random-seed
  # randseed=$((215+10*each_numrun))
  randseed=42

  # : run scripts

  echo -e "python eecs598_pgd.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm $DATANORM\
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA \ 
    --numrun $each_numrun \
    --att-type $att_type \
    --att-tar $att_tar \
    --att-step-size $att_step_size \
    --att-num-steps $att_num_steps \
    --att-epsilon $att_epsilon"


  python eecs598_pgd.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm $DATANORM\
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --numrun $each_numrun \
    --att-type $att_type \
    --att-tar $att_tar \
    --att-step-size $att_step_size \
    --att-num-steps $att_num_steps \
    --att-epsilon $att_epsilon

done
done
# done
