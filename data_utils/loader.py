#!/usr/bin/env python3

"""Data loader."""
import os
import sys
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

# from utils import logging
from data_utils.datasets import *

# logger = logging.get_logger("dam-vp")
_DATASET_CATALOG = {
    ### preparing for meta training
    "sun397": SUN397, 
    "stl10": STL10, 
    "fru92": Fru92Dataset, 
    "veg200": Veg200Dataset, 
    "oxford-iiit-pets": OxfordIIITPet, 
    "eurosat": EuroSAT, 
    ### preparing for task adapting
    "cifar10": CIFAR10, 
    "cifar100": CIFAR100, 
    "cub200": CUB200Dataset,  
    "cub": CUB200Dataset,  
    "nabirds": NabirdsDataset, 
    "oxford-flowers": FlowersDataset, 
    "flower102": FlowersDataset, 
    "stanford-dogs": DogsDataset, 
    "stanford_dogs": DogsDataset, 
    "stanford-cars": CarsDataset, 
    "stanford_cars": CarsDataset, 
    "fgvc-aircraft": AircraftDataset, 
    "food101": Food101, 
    "dtd": DTD, 
    "svhn": SVHN, 
    "gtsrb": GTSRB
}

_DATA_DIR_CATALOG = {
    ### preparing for meta training
    "sun397": "sun397/", 
    "stl10": "stl10/", 
    "fru92": "fru92/", 
    "veg200": "veg200/", 
    "oxford-iiit-pets": "oxford_pets/", 
    "eurosat": "eurosat/", 
    ### preparing for task adapting
    "cifar10": "CIFAR10/", 
    "cifar100": "CIFAR100/", 
    "cub200": "cub/",  
    "cub": "cub/",  
    "nabirds": "nabirds/", 
    "oxford-flowers": "oxford_flowers/", 
    "flower102": "oxford_flowers/", 
    "stanford_dogs": "stanford_dogs/", 
    "stanford_cars": "stanford_cars/", 
    "fgvc-aircraft": "fgvc_aircraft/", 
    "food101": "food-101/", 
    "dtd": "dtd/", 
    "svhn": "svhn/", 
    "gtsrb": "gtsrb/"
}

_NUM_CLASSES_CATALOG = {
    ### preparing for meta training
    "sun397": 397, 
    "stl10": 10, 
    "fru92": 92, 
    "veg200": 200, 
    "oxford-iiit-pets": 37, 
    "eurosat": 10, 
    ### preparing for task adapting
    "cifar10": 10, 
    "cifar100": 100, 
    "cub200": 200,  
    "nabirds": 555, 
    "oxford-flowers": 102, 
    "stanford-dogs": 120, 
    "stanford-cars": 196, 
    "fgvc-aircraft": 100, 
    "food101": 101, 
    "dtd": 47, 
    "svhn": 10, 
    "gtsrb": 43
}


def get_dataset_classes(dataset):
    """Given a dataset, return the name list of dataset classes."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "_class_ids"):
        return dataset._class_ids
    elif hasattr(dataset, "labels"):
        return dataset.labels
    else:
        raise NotImplementedError


def _construct_loader(args, dataset, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = dataset

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        args.data_dir = os.path.join(args.base_dir, "VTAB/")
        from data_utils.datasets.tf_dataset import TFDataset
        dataset = TFDataset(args, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        args.data_dir = os.path.join(args.base_dir, _DATA_DIR_CATALOG[dataset_name])
        dataset = _DATASET_CATALOG[dataset_name](args, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if args.distributed else None

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=drop_last,
    )
    if args.pretrained_model.startswith("clip-"):# and args.adapt_method in ["vp", "ours"]:
        return loader, get_dataset_classes(dataset)
    return loader


def construct_train_loader(args, dataset=None):
    """Train loader wrapper."""
    # if args.distributed:
    #     drop_last = True
    # else:
    #     drop_last = False
    drop_last = False
    return _construct_loader(
        args=args,
        split="train",
        batch_size=int(args.batch_size), 
        shuffle=True,
        drop_last=drop_last,
        dataset=dataset if dataset else args.dataset
    )


def construct_val_loader(args, dataset=None, batch_size=None):
    """Validation loader wrapper."""
    if batch_size is None:
        bs = int(args.batch_size / args.num_gpus)
    else:
        bs = batch_size
    return _construct_loader(
        args=args,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def construct_test_loader(args, dataset=None):
    """Test loader wrapper."""
    return _construct_loader(
        args=args,
        split="test",
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def _dataset_class_num(dataset_name):
    """Query to obtain class nums of datasets."""
    return _NUM_CLASSES_CATALOG[dataset_name]
