#!/usr/bin/env python3

from datetime import datetime
import os
import glob
import sys
import argparse
from colorama import Fore , Style
from torchvision import transforms
from PIL import Image
import yaml

def main():
    parser = argparse.ArgumentParser(description='Create video from dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-cs', '--crop_size', type=int, default=300,
                        help='Size of the random and center crop.')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    synfeal_path = os.getenv('SYNFEAL_DATASET')
    dataset_path = f'{synfeal_path}/datasets/localbot/{args["dataset"]}'
  
    rgb_transform = transforms.Compose([
        transforms.Resize(args['crop_size']),

    ])

    # Check if the appropriate folders exist
    if not os.path.exists(dataset_path):
        print(f'{Fore.RED} Dataset does not exist! {Fore.RESET} Dataset path: {dataset_path}')
        exit(0)


    dataset = sorted(glob.glob(f'{dataset_path}/*.png'))

    for idx ,image in enumerate(dataset):
        rgb_image = Image.open(image)
        rgb_image = rgb_transform(rgb_image)
        filename = f'frame-{idx:05d}'
        print(f'Processed {filename}...')
        rgb_image.save(f'{dataset_path}/{filename}.rgb.png')


if __name__ == "__main__":
    main()