import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from pyrr import Quaternion
from pyrr.quaternion import cross as q_cross, inverse as q_inv
from math import atan2
from localbot_core.src.utilities import poseToMatrix, matrixToRodrigues

def normalize_quat(x, p=2, dim=1):
    """
    Divides a tensor along a certain dim by the Lp norm
    :param x: 
    :param p: Lp norm
    :param dim: Dimension to normalize along
    :return: 
    """
    
    if torch.is_tensor(x):
        # x.shape = (N,4)
        xn = x.norm(p=p, dim=dim) # computes the norm: 1xN
        x = x / xn.unsqueeze(dim=dim)
    
    else: # numpy
        xn = np.linalg.norm(x)
        x = x/xn
        
    return x

def process_pose(pose):
    quat_unit = normalize_quat(pose[:,3:])
    return torch.cat((pose[:,:3], quat_unit), dim=1)


def compute_position_error(pred, targ):
    pred = pred[:3]
    targ = targ[:3]
    
    return mean_squared_error(pred, targ, squared=False) # RMSE

def compute_rotation_error(pred, targ): 
    
    ## first way: using quaternions:
    ## https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames and https://stackoverflow.com/questions/23260939/distance-or-angular-magnitude-between-two-quaternions and https://stackoverflow.com/questions/20798056/magnitude-of-rotation-between-two-quaternions
    #pred = pred[3:]
    #targ = targ[3:]
    # pred = Quaternion(pred)
    # targ = Quaternion(targ)
    # deltaq = q_cross(pred, q_inv(targ))
    # return 2 * atan2(np.linalg.norm(deltaq[:3]), deltaq[3]) # angle in rads
    
    ## second way: using rodrigues (like ATOM) --> better because angle ranges from 0 to pi (whereas with quaterions ranges from 0 to 2pi)
    ## https://github.com/lardemua/atom/blob/284b7943e467e53a3258de6f673cf852b07654cb/atom_evaluation/scripts/camera_to_camera_evalutation.py#L290
    pred_matrix = poseToMatrix(pred)
    targ_matrix = poseToMatrix(targ)
    
    delta = np.dot(np.linalg.inv(pred_matrix), targ_matrix)
    deltaR = matrixToRodrigues(delta[0:3, 0:3])
    
    return np.linalg.norm(deltaR)
    
    
    
    
    
    