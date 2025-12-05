#!/bin/bash

# Configuration for Oxford Pets dataset
n_epochs=100
warmup_epochs=20
loader="DAM-VP"
dataset="oxford-iiit-pets"
batch_size=64
scheduler="cosine"
optimizer="AdamW"
p_len=16

# Model parameters
cs=("0.001")
ds=("0.5")
trainer="ours"
mixup="mixup"
transform="default"
net=19
prompt_patch=16
hid_dim=16

# Kernel parameters
as=("7")
bs=("9")
k3="3"
deep_layer=12
type="ours9"
cutmix_alpha="1.0"
bn="none"
prompt_patch_2=11
prompt_patch_22=25
simam=False

# Data path - adjust this to your dataset location
# The dataset should be at: base_dir/torchvision_dataset/oxford-iiit-pet/
# We create a symlink from oxford_pets to oxford-iiit-pet if needed
BASE_DIR="/home/user/Documents/data"
# Create symlink if it doesn't exist
if [ ! -d "${BASE_DIR}/torchvision_dataset/oxford-iiit-pet" ]; then
    mkdir -p "${BASE_DIR}/torchvision_dataset"
    ln -sf "${BASE_DIR}/oxford_pets" "${BASE_DIR}/torchvision_dataset/oxford-iiit-pet"
    echo "Created symlink: ${BASE_DIR}/torchvision_dataset/oxford-iiit-pet -> ${BASE_DIR}/oxford_pets"
fi

# Save mask every N iterations (set to 0 to disable)
SAVE_MASK_INTERVAL=100

for k1 in "${as[@]}"
do
    for k2 in "${bs[@]}"
    do
        for lr in "${cs[@]}"
        do
            for wd in "${ds[@]}"
            do
                python main.py --info="InstanceVPD-${net}-n=${prompt_patch}-h=${hid_dim}-n2=${prompt_patch_2}-n22=${prompt_patch_22}-w=2-lr=${lr}-bs=${batch_size}-dl=${deep_layer}-mixup=${mixup}-${cutmix_alpha}-drop-${k1}-${k2}-${k3}" \
                        --model=InstanceVPD  \
                        --output_path="./Output/${dataset}/ViT/imagenet22k/InsVP" \
                        --n_epochs=$n_epochs --meta_net=$net --hid_dim=$hid_dim --prompt_patch=$prompt_patch \
                        --pretrained=imagenet22k --prompts_2_weight=2 \
                        --batch_size=$batch_size \
                        --prompt_patch_2=$prompt_patch_2 --prompt_patch_22=$prompt_patch_22 \
                        --dataset=$dataset \
                        --lr=$lr --scheduler=$scheduler \
                        --weight_decay=$wd --optimizer=$optimizer \
                        --loader=$loader \
                        --transform=$transform \
                        --deep_prompt_type=${type} \
                        --deep_layer=${deep_layer} \
                        --mixup=${mixup} \
                        --cutmix_alpha=${cutmix_alpha} \
                        --TP_kernel_1=${k1} \
                        --TP_kernel_2=${k2} \
                        --TP_kernel_3=${k3} \
                        --p_len=${p_len} \
                        --p_len_vpt=${p_len} \
                        --trainer=${trainer} \
                        --simam=${simam} \
                        --warmup_epochs=$warmup_epochs \
                        --base_dir="${BASE_DIR}" \
                        --save_mask_interval=${SAVE_MASK_INTERVAL}
            done
        done
    done
done

echo "Training completed! Masks are saved in ./Output/${dataset}/ViT/imagenet22k/InsVP/<timestamp>/masks/"
