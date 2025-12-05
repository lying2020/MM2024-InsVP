# Desc


import torch
import torchvision
from models import get_model
import time
from train.utils import device, get_dataset, cosine_lr
from train.scheduler import WarmupCosineSchedule as cosine_lr_2
import torch.optim as optim
import torch.nn as nn
import logging
import os
from torch.amp import autocast, GradScaler

from models.utils import Dataset_N_classes
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import numpy as np
from torchvision.utils import save_image




def train(args):
    """
    The training process
    """
    if args.loader == 'DAM-VP':
        from data_utils import loader as data_loader
    elif args.loader == 'E2VPT':
        from src.data import loader as data_loader
    logger_path = os.path.join(args.output_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log')
    logging.basicConfig(filename=logger_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create directory for saving masks
    mask_save_dir = os.path.join(args.output_path, 'masks')
    os.makedirs(mask_save_dir, exist_ok=True)

    model = get_model(args)
    model.mode = 'train'

    scaler = GradScaler()

    if args.mixup == 'mixup':
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()


    if args.optimizer == 'SGD':
        optimizer = model.get_optimizer()
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.learnable_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    try:
        train_loader = data_loader.construct_train_loader(args, args.dataset)
    except:
        train_loader = data_loader.construct_train_loader(args)

    if args.scheduler == 'cosine-2':
        scheduler = cosine_lr_2(optimizer,
            len(train_loader)*args.warmup_epochs,
            len(train_loader)*args.n_epochs
        )
    elif args.scheduler == 'cosine':
            scheduler = cosine_lr(optimizer, args.lr,
                    len(train_loader)*args.n_epochs//5,
                    len(train_loader)*args.n_epochs
                )

    if args.mixup == 'mixup':
        mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_minmax=None,
        prob=0.5,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=Dataset_N_classes[args.dataset]
        )

    accs = []
    best_acc = 0.0
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            if args.scheduler == 'cosine':
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
            try:
                images, labels = data['image'], data['label']
            except:
                images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if args.mixup == 'mixup':
                if images.size(0) % 2 == 1:
                    images = images[:-1]
                    labels = labels[:-1]
                images, labels = mixup_fn(images, labels)

            with autocast(device_type='cuda'):
                # Get outputs and masks
                if i % args.save_mask_interval == 0 and args.save_mask_interval > 0:
                    outputs, masks = model(images, return_mask=True)
                else:
                    outputs = model(images)
                    masks = None

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            try:
                correct += (predicted == labels).sum().item()
            except:
                ...
            running_loss += loss.item()

            # Save masks periodically
            if masks is not None:
                # Save masks as images
                mask_save_path = os.path.join(mask_save_dir, f'epoch_{epoch+1}_iter_{i}_mask.png')
                # Normalize mask to [0, 1] for visualization
                mask_vis = (masks - masks.min()) / (masks.max() - masks.min() + 1e-8)
                save_image(mask_vis[:min(8, masks.size(0))], mask_save_path, nrow=4)

                # Also save as numpy array for further analysis
                mask_np_path = os.path.join(mask_save_dir, f'epoch_{epoch+1}_iter_{i}_mask.npy')
                np.save(mask_np_path, masks.detach().cpu().numpy())

                print(f"Saved mask at epoch {epoch+1}, iteration {i}")
                logging.info(f"Saved mask at epoch {epoch+1}, iteration {i}")

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
            logging.info(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        print(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        logging.critical(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")

        acc = eval(model, args, epoch=epoch)
        accs.append(acc)

        if args.scheduler == 'cosine-2':
            scheduler.step()

        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(args.output_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        print(f"Best accuracy: {best_acc:.4f}")
        logging.critical(f"Best accuracy: {best_acc:.4f}")
        accs_tmp = [round(acc, 4) for acc in accs]
        print(f"All accuracy: {accs_tmp}")
        logging.critical(f"All accuracy: {accs_tmp}")
        best_acc_tmp = round(best_acc, 4)
        with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
            f.write(f"Best accuracy:\n {best_acc_tmp}\n")
            f.write(f"All accuracy:\n {accs_tmp}\n")

    best_acc = round(best_acc, 4)
    accs = [round(acc, 4) for acc in accs]

    print("Finished Training")
    logging.critical("Finished Training")
    print(f"Best accuracy: {best_acc}")
    logging.critical(f"Best accuracy: {best_acc}")
    print(f"All accuracy: {accs}")
    logging.critical(f"All accuracy: {accs}")
    with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
        f.write(f"Best accuracy:\n {best_acc}\n")
        f.write(f"All accuracy:\n {accs}\n")



def eval(model, args, epoch=0, save_mask=False):
    """
    The evaluation process
    """
    if args.loader == 'DAM-VP':
        from data_utils import loader as data_loader
    elif args.loader == 'E2VPT':
        from src.data import loader as data_loader
    model.mode = 'test'
    model.eval()
    try:
        test_loader = data_loader.construct_test_loader(args, args.dataset)
    except:
        test_loader = data_loader.construct_test_loader(args)

    # Create directory for saving masks if needed
    mask_save_dir = None
    if save_mask:
        mask_save_dir = os.path.join(args.output_path, 'test_masks')
        os.makedirs(mask_save_dir, exist_ok=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            try:
                images, labels = data['image'], data['label']
            except:
                images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # Get outputs and masks if saving masks
            if save_mask and (i % args.save_mask_interval == 0 if args.save_mask_interval > 0 else True):
                outputs, masks = model(images, return_mask=True)
            else:
                outputs = model(images)
                masks = None
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save masks if enabled
            if masks is not None and mask_save_dir is not None:
                # Save masks as images
                mask_save_path = os.path.join(mask_save_dir, f'test_iter_{i}_mask.png')
                # Normalize mask to [0, 1] for visualization
                mask_vis = (masks - masks.min()) / (masks.max() - masks.min() + 1e-8)
                save_image(mask_vis[:min(8, masks.size(0))], mask_save_path, nrow=4)
                
                # Also save as numpy array for further analysis
                mask_np_path = os.path.join(mask_save_dir, f'test_iter_{i}_mask.npy')
                np.save(mask_np_path, masks.detach().cpu().numpy())
                
                print(f"Saved test mask at iteration {i}")
                logging.info(f"Saved test mask at iteration {i}")
    
    accuracy = correct / total
    print(f"Accuracy of the network on the {total} test images:\n {accuracy:.4f}")
    logging.critical(f"Accuracy of the network on the {total} test images:\n {accuracy:.4f}")
    model.mode = 'train'
    return accuracy


def test(args):
    """
    The testing process - load model and evaluate
    """
    if args.loader == 'DAM-VP':
        from data_utils import loader as data_loader
    elif args.loader == 'E2VPT':
        from src.data import loader as data_loader
    
    logger_path = os.path.join(args.output_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'_test.log')  
    logging.basicConfig(filename=logger_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load model
    model = get_model(args)
    model.mode = 'test'
    
    # Load checkpoint if provided
    if args.model_load_path:
        if os.path.exists(args.model_load_path):
            print(f"Loading model from {args.model_load_path}")
            logging.info(f"Loading model from {args.model_load_path}")
            checkpoint = torch.load(args.model_load_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully!")
            logging.info("Model loaded successfully!")
        else:
            print(f"Warning: Model path {args.model_load_path} does not exist!")
            logging.warning(f"Model path {args.model_load_path} does not exist!")
    else:
        print("Warning: No model_load_path provided. Using randomly initialized model.")
        logging.warning("No model_load_path provided. Using randomly initialized model.")
    
    # Run evaluation with mask saving if enabled
    save_mask = args.save_mask_interval > 0
    accuracy = eval(model, args, epoch=0, save_mask=save_mask)
    
    # Save test results
    with open(os.path.join(args.output_path, 'test_accuracy.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Model Path: {args.model_load_path}\n")
    
    print(f"Test completed! Accuracy: {accuracy:.4f}")
    if save_mask:
        print(f"Masks saved in: {os.path.join(args.output_path, 'test_masks')}")
    logging.critical(f"Test completed! Accuracy: {accuracy:.4f}")
