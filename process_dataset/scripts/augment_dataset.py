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
    parser.add_argument('-o', '--output', type=str, required=True, help='Name of the output dataset')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    synfeal_path = os.getenv('SYNFEAL_DATASET')
    dataset_path = f'{synfeal_path}/datasets/localbot/{args["dataset"]}'
    output_path = f'{synfeal_path}/datasets/localbot/{args["output"]}'
  
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomErasing()
        transforms.ColorJitter(brightness=.5, hue=0)
    ])

    tensor_to_pil = transforms.ToPILImage()

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
            os.system('mkdir -p' + output_path)
        if ans.lower() not in ['', 'yes', 'y']: # If the user does not want to resume training
            print(f'{Fore.RED} Terminating dataset augmentation... {Fore.RESET}')
            exit(0)
    else:
        os.system('mkdir -p ' + output_path)

    with open(f'{dataset_path}/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    with open(f'{dataset_path}/model3d_config.yaml') as f:
        model3d_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    dt_now = datetime.now() # current date and time
    config['date'] = dt_now.strftime("%d/%m/%Y, %H:%M:%S")
    config['depth']['intrinsic'] = f'{output_path}/depth_intrinsic.txt'
    config['rgb']['intrinsic'] = f'{output_path}/rgb_intrinsic.txt'

    with open(f'{output_path}/config.yaml', 'a') as file:
        yaml.dump(config, file)
    
    with open(f'{output_path}/model3d_config.yaml', 'a') as file:
        yaml.dump(model3d_config, file)

    
    # open both files
    with open(f'{dataset_path}/rgb_intrinsic.txt','r') as firstfile, open(f'{output_path}/rgb_intrinsic.txt','a') as secondfile:
        # read content from first file
        for line in firstfile:
            # append content to second file
            secondfile.write(line)

    with open(f'{dataset_path}/depth_intrinsic.txt','r') as firstfile, open(f'{output_path}/depth_intrinsic.txt','a') as secondfile:
        # read content from first file
        for line in firstfile:
            # append content to second file
            secondfile.write(line)


    dataset = sorted(glob.glob(f'{dataset_path}/*.png'))

    for idx ,image in enumerate(dataset):
        rgb_image = Image.open(image)
        rgb_image = rgb_transform(rgb_image)
        rgb_image = tensor_to_pil(rgb_image)
        filename = f'frame-{idx:05d}'
        print(f'Processed {filename}...')
        rgb_image.save(f'{output_path}/{filename}.rgb.png')
        with open(f'{dataset_path}/{filename}.pose.txt','r') as firstfile, open(f'{output_path}/{filename}.pose.txt','a') as secondfile:
            # read content from first file
            for line in firstfile:
                # append content to second file
                secondfile.write(line)


if __name__ == "__main__":
    main()