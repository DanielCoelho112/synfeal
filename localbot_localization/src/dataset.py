import cv2
from cv2 import INTER_AREA
import torch.utils.data as data
from localbot_core.src.utilities import read_pcd, matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os
import yaml
from yaml.loader import SafeLoader
from PIL import Image
from torchvision import transforms

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class LocalBotDataset(data.Dataset):
    def __init__(self, path_seq, rgb_transform = None, depth_transform = None, inputs = None):
        self.root = f'{os.environ["HOME"]}/datasets/localbot'
        self.seq = path_seq
        self.path_seq = f'{self.root}/{path_seq}'
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        if inputs == None:
            self.inputs = ['point_cloud', 'depth_image', 'rgb_image']
        else:
            self.inputs = inputs
        
        config = self.getConfig()
        if 'statistics' in config:
            self.depth_mean = config['statistics']['D']['mean']
            self.depth_std = config['statistics']['D']['std']
        
        
    def __getitem__(self, index):
        
        output = []
        
        if 'point_cloud' in self.inputs:
            # load point cloud
            pc_raw = read_pcd(f'{self.path_seq}/frame-{index:05d}.pcd')
            point_set = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
            point_set = torch.from_numpy(point_set.astype(np.float32))
            output.append(point_set)
        
        if 'depth_image' in self.inputs:
            # load depth image
            depth_image = cv2.imread(f'{self.path_seq}/frame-{index:05d}.depth.png', cv2.IMREAD_UNCHANGED)
            depth_image = depth_image.astype(np.float32) / 1000.0  # to meters
            
            # do cv show to see if we have the same image!
            if self.depth_transform['normalize']:
                depth_image = (depth_image - self.depth_mean) /  self.depth_std
            
            if 'resize' in self.depth_transform:
                # resize image
                depth_image = cv2.resize(depth_image, dsize=(self.depth_transform['resize'][0],self.depth_transform['resize'][1]), interpolation=INTER_AREA)
            h,w = depth_image.shape
            depth_image = depth_image.reshape((1,h,w))
            depth_image = torch.from_numpy(depth_image)
            output.append(depth_image)
        
        if 'rgb_image' in self.inputs:
            # TODO: change this to the correct dataset
            rgb_image = Image.open(f'{self.path_seq}/frame-{index:05d}.rgb.png')
            if self.rgb_transform != None:
                rgb_image = self.rgb_transform(rgb_image)
            output.append(rgb_image)

        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        quaternion = matrixToQuaternion(matrix)
        quaternion = normalize_quat(quaternion)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, quaternion)
        pose = torch.from_numpy(pose.astype(np.float32))
        output.append(pose)
        
        return tuple(output)
    

    def __len__(self):
        return sum(f.endswith('pose.txt') for f in os.listdir(self.path_seq))
    
    def getConfig(self):
        with open(f'{self.path_seq}/config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        return config

    def setConfig(self, config):
        with open(f'{self.path_seq}/config.yaml', 'w') as f:
            yaml.dump(config, f)
        
# transform = transforms.Compose([
#         transforms.Resize(300),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#depth_transform= {'resize':[299,299], 'normalize' : True}

#dataset = LocalBotDataset('seq4dpii_v2',depth_transform=depth_transform ,rgb_transform=transform, inputs=['point_cloud','rgb_image'])
#print(dataset[1][1].shape)
#cv2.imshow(dataset[0][2])
#cv2.waitKey(0)
#print(sum(torch.isnan(dataset[78][0])))

