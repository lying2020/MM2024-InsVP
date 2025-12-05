#!/bin/bash

# Test script for Oxford Pets dataset
# This script loads a trained model and evaluates it on the test set

# Configuration
loader="DAM-VP"
dataset="oxford-iiit-pets"
batch_size=64

# Model parameters (should match training parameters)
net=19
prompt_patch=16
hid_dim=16
prompt_patch_2=11
prompt_patch_22=25
deep_layer=12
type="ours9"
p_len=16
simam=False

# Data path
BASE_DIR="/home/user/Documents/data"

# Model checkpoint path - CHANGE THIS to your trained model path
MODEL_PATH="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/<timestamp>/best_model.pth"

# Save mask every N iterations (set to 0 to disable, or a number like 10 to save every 10 batches)
SAVE_MASK_INTERVAL=10

# Create symlink if it doesn't exist
if [ ! -d "${BASE_DIR}/torchvision_dataset/oxford-iiit-pet" ]; then
    mkdir -p "${BASE_DIR}/torchvision_dataset"
    ln -sf "${BASE_DIR}/oxford_pets" "${BASE_DIR}/torchvision_dataset/oxford-iiit-pet"
    echo "Created symlink: ${BASE_DIR}/torchvision_dataset/oxford-iiit-pet -> ${BASE_DIR}/oxford_pets"
fi

# Run test
python main.py \
    --mode=test \
    --info="Test-OxfordPets" \
    --model=InstanceVPD \
    --output_path="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test" \
    --meta_net=$net \
    --hid_dim=$hid_dim \
    --prompt_patch=$prompt_patch \
    --pretrained=imagenet22k \
    --prompts_2_weight=2 \
    --batch_size=$batch_size \
    --prompt_patch_2=$prompt_patch_2 \
    --prompt_patch_22=$prompt_patch_22 \
    --dataset=$dataset \
    --loader=$loader \
    --transform=default \
    --deep_prompt_type=${type} \
    --deep_layer=${deep_layer} \
    --TP_kernel_1=7 \
    --TP_kernel_2=9 \
    --TP_kernel_3=3 \
    --p_len=${p_len} \
    --p_len_vpt=${p_len} \
    --trainer=ours \
    --simam=${simam} \
    --base_dir="${BASE_DIR}" \
    --model_load_path="${MODEL_PATH}" \
    --save_mask_interval=${SAVE_MASK_INTERVAL}

echo "Test completed! Results saved in ./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test/"
if [ ${SAVE_MASK_INTERVAL} -gt 0 ]; then
    echo "Masks saved in ./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test/test_masks/"
fi

