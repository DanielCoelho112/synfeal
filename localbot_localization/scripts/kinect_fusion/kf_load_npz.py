import numpy as np


# poses = np.load('/home/danc/datasets/localbot/processed1/raw_poses.npz')
# print(poses['c2w_mats'][-1])

# cams = np.load('/home/danc/datasets/localbot/processed1/cameras.npz')
# print(cams['world_mats'][0])


poses = np.load('/home/danc/Desktop/KinectFusion/reconstruct/fr1_localbot13/traj_gt.npz')
print(poses['poses'][1])