import torch.utils.data as data
from localbot_core.src.utilities import read_pcd, matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class LocalBotDataset(data.Dataset):
    def __init__(self, path_seq, npoints, scaling = False):
        self.path_seq = f'{os.environ["HOME"]}/datasets/localbot/{path_seq}'
        self.npoints = npoints # TODO: remove this because in this phase all point clouds should be downsampled.
        self.scaling = scaling  # TODO: remove this because this was done in the dataset validation!
        self.nframes = sum(f.endswith('.txt') for f in os.listdir(self.path_seq))
        

    def __getitem__(self, index):
        # return Nx3, 1x6 torch tensors
        
        # load point cloud
        pc_raw = read_pcd(f'{self.path_seq}/frame-{index:05d}.pcd')
        pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
        
        # remove nan points
        pts = pts[~np.isnan(pts).any(axis=1)]  # TODO: remove this
        

        choice = np.random.choice(len(pts), self.npoints, replace=True) # list with the indexes  # TODO remove this
        point_set = pts[choice, :] # downsampling # TODO: remove this
        
        
        # TODO: should we scale our point cloud? Usually, this is good for neural networks, however, in our case I think we are losing valuable information...
        #       try both with and without
        if self.scaling:  # TODO: remove this!!
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist
        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        quaternion = matrixToQuaternion(matrix)
        quaternion = normalize_quat(quaternion)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, quaternion)
        
        point_set = torch.from_numpy(point_set.astype(np.float32))
        pose = torch.from_numpy(pose.astype(np.float32))
        
        return point_set, pose
                

    def __len__(self):
        return self.nframes



dataset = LocalBotDataset('seq110', 100000)
print(dataset[78][1])
# #print(sum(torch.isnan(dataset[78][0])))

