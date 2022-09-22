#!/usr/bin/env python3

# stdlib
import sys
import argparse
import copy

# 3rd-party
from dataset import LocalBotDataset
import os
import shutil
import yaml

def main():    
    parser = argparse.ArgumentParser(description='Validate dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-dr', '--dataset_reduced', type=str, required=True, help='Suffix to append to the name of the dataset')
    parser.add_argument('-s', '--size', type=int, required=True, help='Sample size')
   
    
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    dataset = LocalBotDataset(path_seq=args['dataset'])
    path_root = dataset.root
    
    dataset_reduced_path = f'{path_root}/{args["dataset_reduced"]}'

    
    if os.path.exists(dataset_reduced_path):
        print(f'{dataset_reduced_path} already exits. Aborting reducing')
        exit(0)
    else:
        os.makedirs(dataset_reduced_path)  # Create the new folder

    # get config
    config = dataset.getConfig()
    if 'statistics' in config:
        config.pop('statistics')
    
    if not config['fast']:
        files_to_copy = ['.pcd', '.rgb.png', '.depth.png','.pose.txt']
    else:
        files_to_copy = ['.rgb.png','.pose.txt']
    
    # copy intrinsics to both datasets
    
    for idx in range(len(dataset)):
        print(f'original idx: {idx}')
        
        if idx <= args['size']:
            print(f'copying {idx} to {idx} in {dataset_reduced_path}')
            for file in files_to_copy:
                shutil.copy2(f'{dataset.path_seq}/frame-{idx:05d}{file}', f'{dataset_reduced_path}/frame-{idx:05d}{file}')
       
    # copy intrinsics to both datasets
    shutil.copy2(f'{dataset.path_seq}/depth_intrinsic.txt', f'{dataset_reduced_path}/depth_intrinsic.txt')  
    shutil.copy2(f'{dataset.path_seq}/rgb_intrinsic.txt', f'{dataset_reduced_path}/rgb_intrinsic.txt')  
        
    config['raw'] = args['dataset_reduced']
    with open(f'{dataset_reduced_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)
    
        
if __name__ == "__main__":
    main()



