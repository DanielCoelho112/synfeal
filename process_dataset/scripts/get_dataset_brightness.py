#!/usr/bin/env python3

import os
import glob
import sys
import argparse
from colorama import Fore , Style
import cv2 
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Create video from dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    synfeal_path = os.getenv('SYNFEAL_DATASET')
    dataset_path = f'{synfeal_path}/datasets/localbot/{args["dataset"]}'

    # Check if the appropriate folders exist
    if not os.path.exists(dataset_path):
        print(f'{Fore.RED} Dataset does not exist! {Fore.RESET} Dataset path: {dataset_path}')
        exit(0)

    dataset = sorted(glob.glob(f'{dataset_path}/*.png'))

    max_brightness = 0
    min_brightness = 255

    for idx ,image in enumerate(dataset):
        image = cv2.imread(image)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the HSV image into separate channels
        h, s, v = cv2.split(image_hsv)

        # Calculate the average brightness (value channel)
        average_brightness = np.mean(v)

        if average_brightness > max_brightness:
            max_brightness = average_brightness
        
        if average_brightness < min_brightness:
            min_brightness = average_brightness

        print(f'Processed image {idx+1}')

    print(f'{Fore.BLUE}Max brightness: {max_brightness} Min brightness: {min_brightness}{Fore.RESET}')
      


if __name__ == "__main__":
    main()