import numpy as np
import torch

def normalize_quat(x, p=2, dim=1):
    """
    Divides a tensor along a certain dim by the Lp norm
    :param x: 
    :param p: Lp norm
    :param dim: Dimension to normalize along
    :return: 
    """
    # x.shape = (N,4)
    xn = x.norm(p=p, dim=dim) # computes the norm: 1xN
    x = x / xn.unsqueeze(dim=dim)
    return x

def process_pose(pose):
    quat_unit = normalize_quat(pose[:,3:])
    return torch.cat((pose[:,:3], quat_unit), dim=1)