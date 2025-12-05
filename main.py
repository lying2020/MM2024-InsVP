#! /usr/bin/env python3

import importlib
import os
import torch

from argparse import ArgumentParser
from utils.seed import set_random_seed
from utils.args import add_experiment_args, add_management_args, add_model_args, save_args

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

def main():

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description='perturbation', allow_abbrev=False)
    # print("main.py")
    parser.add_argument('--model', type=str,
                        # required=True,
                        default='InstanceVPD',
                        help='Model name.', choices=['InstanceVPD'])

    args = parser.parse_known_args()[0]

    # add arguments for the specific model
    # mod = importlib.import_module('models.' + args.model)

    # get_parser = getattr(mod, 'get_parser')
    # parser = get_parser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    
    # Set default values if not provided
    if not args.base_dir:
        args.base_dir = '/home/liying/Documents/dataset/VLMDataset'
    if not args.data_dir:
        # Construct data_dir from base_dir and dataset
        from data_utils.loader import _DATA_DIR_CATALOG
        if args.dataset in _DATA_DIR_CATALOG:
            args.data_dir = os.path.join(args.base_dir, _DATA_DIR_CATALOG[args.dataset])

    if args.seed is not None:
        print("Setting random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    args = save_args(args)

    if args.trainer == "ours":
        from train import trainer

    # print(args.mode)
    # input()

    if args.mode == 'train':
        trainer.train(args)
    elif args.mode == 'test' or args.mode == 'eval':
        trainer.test(args)

if __name__ == '__main__':
    # print("main.py")
    main()
