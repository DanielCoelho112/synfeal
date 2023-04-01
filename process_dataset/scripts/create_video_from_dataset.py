#!/usr/bin/env python3

import os
import glob
import sys
import argparse
from colorama import Fore , Style
import cv2

def main():
    parser = argparse.ArgumentParser(description='Create video from dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-o', '--output', type=str, required=True, help='Name of the output video')
    parser.add_argument('-f', '--fps', type=int, required=False, default=60, help='Frames per second')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    synfeal_path = os.getenv('SYNFEAL_DATASET')
    dataset_path = f'{synfeal_path}/datasets/localbot/{args["dataset"]}'
    output_path = f'{synfeal_path}/videos/localbot/{args["output"]}'

    # Check if the appropriate folders exist
    if not os.path.exists(dataset_path):
        print(f'{Fore.RED} Dataset does not exist! {Fore.RESET} Dataset path: {dataset_path}')
        exit(0)

    if os.path.exists(output_path):
        print(Fore.YELLOW + f'Folder already exists! Do you want to overwrite?' + Style.RESET_ALL)
        ans = input(Fore.YELLOW + "Y" + Style.RESET_ALL + "ES/" + Fore.YELLOW + "n" + Style.RESET_ALL + "o:") # Asks the user if they want to overwrite the folder
        
        if ans.lower() in ['', 'yes','y']:
            print(Fore.YELLOW + 'Overwriting.' + Style.RESET_ALL)
            os.system('rm -rf ' + output_path)
            os.system('mkdir -p ' + output_path)
        if ans.lower() not in ['', 'yes', 'y']: # If the user does not want to resume training
            print(f'{Fore.RED} Terminating generating video... {Fore.RESET}')
            exit(0)
    else:
        os.system('mkdir -p ' + output_path)

    dataset = sorted(glob.glob(f'{dataset_path}/*.png'))

    frame = cv2.imread(dataset[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f'{output_path}/{args["output"]}.avi', 0, args['fps'], (width,height))

    for idx ,image in enumerate(dataset):
        print(f'Adding image {idx} to video')
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print(f'{Fore.GREEN}Video Saved{Style.RESET_ALL}, at {output_path}')


if __name__ == "__main__":
    main()