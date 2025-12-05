# 运行 Oxford Pets 数据集训练指南

## 已完成的修改

1. **预训练模型路径修改** (`models/utils.py`)
   - 自动检测当前目录下的 `vit_base_p16_224_in22k.pth` 文件
   - 如果不存在，则回退到原始路径

2. **添加保存 Mask 功能** (`train/trainer.py`, `models/InstanceVPD.py`)
   - 模型 forward 方法支持返回 mask
   - 训练过程中定期保存 mask 为图像和 numpy 文件
   - 可通过 `--save_mask_interval` 参数控制保存频率

3. **创建运行脚本** (`sh/oxford_pets.sh`)
   - 配置了 Oxford Pets 数据集的训练参数
   - 自动创建数据集路径的符号链接

## 运行步骤

### 1. 确保数据集路径正确
数据集应该在 `/home/user/Documents/data/oxford_pets/`，包含：
- `images/` 目录
- `annotations/` 目录

### 2. 确保预训练模型存在
预训练模型应该在项目根目录：`vit_base_p16_224_in22k.pth`

### 3. 运行训练脚本

```bash
cd /home/user/MM2024-InsVP
bash sh/oxford_pets.sh
```

或者直接使用 Python 命令：

```bash
python main.py \
    --info="InstanceVPD-OxfordPets" \
    --model=InstanceVPD \
    --output_path="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP" \
    --n_epochs=100 \
    --meta_net=19 \
    --hid_dim=16 \
    --prompt_patch=16 \
    --pretrained=imagenet22k \
    --prompts_2_weight=2 \
    --batch_size=64 \
    --prompt_patch_2=11 \
    --prompt_patch_22=25 \
    --dataset=oxford-iiit-pets \
    --lr=0.001 \
    --scheduler=cosine \
    --weight_decay=0.5 \
    --optimizer=AdamW \
    --loader=DAM-VP \
    --transform=default \
    --deep_prompt_type=ours9 \
    --deep_layer=12 \
    --mixup=mixup \
    --cutmix_alpha=1.0 \
    --TP_kernel_1=7 \
    --TP_kernel_2=9 \
    --TP_kernel_3=3 \
    --p_len=16 \
    --p_len_vpt=16 \
    --trainer=ours \
    --warmup_epochs=20 \
    --base_dir="/home/user/Documents/data" \
    --save_mask_interval=100
```

## 输出结果

训练过程中会生成以下内容：

1. **模型检查点**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/<timestamp>/best_model.pth`
2. **训练日志**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/<timestamp>/*.log`
3. **Mask 文件**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/<timestamp>/masks/`
   - `epoch_X_iter_Y_mask.png` - 可视化的 mask 图像
   - `epoch_X_iter_Y_mask.npy` - mask 的 numpy 数组（可用于进一步分析）

## 参数说明

- `--save_mask_interval`: 每 N 次迭代保存一次 mask（默认 100，设为 0 可禁用）
- `--base_dir`: 数据集根目录，脚本会自动创建符号链接
- `--n_epochs`: 训练轮数
- `--batch_size`: 批次大小

## 测试模式（仅测试，不训练）

如果你已经有一个训练好的模型，只想进行测试并保存 mask，可以使用测试模式：

### 1. 准备训练好的模型

确保你有训练好的模型检查点文件（通常是 `best_model.pth`）

### 2. 运行测试脚本

```bash
cd /home/user/MM2024-InsVP
bash sh/test_oxford_pets.sh
```

**重要**: 在运行前，需要修改 `sh/test_oxford_pets.sh` 中的 `MODEL_PATH` 变量，指向你的模型文件路径。

例如：
```bash
MODEL_PATH="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/2024-01-01-12-00-00-InstanceVPD/best_model.pth"
```

### 3. 或者直接使用 Python 命令

```bash
python main.py \
    --mode=test \
    --info="Test-OxfordPets" \
    --model=InstanceVPD \
    --output_path="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test" \
    --meta_net=19 \
    --hid_dim=16 \
    --prompt_patch=16 \
    --pretrained=imagenet22k \
    --prompts_2_weight=2 \
    --batch_size=64 \
    --prompt_patch_2=11 \
    --prompt_patch_22=25 \
    --dataset=oxford-iiit-pets \
    --loader=DAM-VP \
    --transform=default \
    --deep_prompt_type=ours9 \
    --deep_layer=12 \
    --TP_kernel_1=7 \
    --TP_kernel_2=9 \
    --TP_kernel_3=3 \
    --p_len=16 \
    --p_len_vpt=16 \
    --trainer=ours \
    --simam=False \
    --base_dir="/home/user/Documents/data" \
    --model_load_path="./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/<timestamp>/best_model.pth" \
    --save_mask_interval=10
```

### 测试输出结果

测试过程中会生成：

1. **测试日志**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test/<timestamp>_test.log`
2. **测试准确率**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test/test_accuracy.txt`
3. **Mask 文件**: `./Output/oxford-iiit-pets/ViT/imagenet22k/InsVP/test/test_masks/`
   - `test_iter_X_mask.png` - 可视化的 mask 图像
   - `test_iter_X_mask.npy` - mask 的 numpy 数组

## 注意事项

1. 确保有足够的 GPU 内存（建议至少 8GB）
2. Mask 保存会占用额外的磁盘空间
3. 如果数据集路径不同，请修改脚本中的 `BASE_DIR` 变量
4. **测试模式**：使用 `--mode=test` 或 `--mode=eval`，必须提供 `--model_load_path` 参数
5. **训练模式**：使用 `--mode=train`（默认值）
