import torch.utils.data as data
from localbot_core.src.utilities import *
import numpy as np
import torch

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class LocalBotDataset(data.Dataset):
    def __init__(self, seq, npoints, scaling = False):
        self.path_seq = f'/home/danc/datasets/localbot/seq{seq}'
        self.npoints = npoints # lets use 100 000 for now!
        self.scaling = scaling
        self.nframes = sum(f.endswith('.txt') for f in os.listdir(f'/home/danc/datasets/localbot/seq{seq}'))
        

    def __getitem__(self, index):
        # return Nx3, 1x6 torch tensors
        
        # load point cloud
        pc_raw = read_pcd(f'{self.path_seq}/frame-{index:05d}.pcd')
        pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
        
        # remove nan points
        pts = pts[~np.isnan(pts).any(axis=1)]
        

        choice = np.random.choice(len(pts), self.npoints, replace=True) # list with the indexes
        point_set = pts[choice, :] # downsampling
        
        
        # TODO: should we scale our point cloud? Usually, this is good for neural networks, however, in our case I think we are losing valuable information...
        #       try both with and without
        if self.scaling:  
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist
        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        rodrigues = matrixToRodrigues(matrix)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, rodrigues)
        
        point_set = torch.from_numpy(point_set.astype(np.float32))
        pose = torch.from_numpy(pose.astype(np.float32))
        
        return point_set, pose
                

    def __len__(self):
        return self.nframes



dataset = LocalBotDataset(110, 100000)
print(dataset[78][0].shape)
#print(sum(torch.isnan(dataset[78][0])))

