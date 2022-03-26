import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import atan
from pyrr import Quaternion
from pyrr.quaternion import cross as q_cross, inverse as q_inv
from math import atan2

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

def quaternion_multiply(Q0,Q1): # https://automaticaddison.com/how-to-multiply-two-quaternions-together-using-python/
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
     
    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
     
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion # distance in meters


def compute_position_error(pred, targ):
    pred = pred[:3]
    targ = targ[:3]
    
    return mean_squared_error(pred, targ, squared=False) # RMSE

def compute_rotation_error(pred, targ): # https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames
    pred = pred[3:]
    targ = targ[3:]
    
    pred = Quaternion(pred)
    targ = Quaternion(targ)
    
    deltaq = q_cross(pred, q_inv(targ))
    
    return atan2(np.linalg.norm(deltaq[:3]), deltaq[3]) # angle in rads
        
    
    
    