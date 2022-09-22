import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import cv2
from synfeal_collection.src.pypcd_no_ros import PointCloud
import torch
from sklearn.metrics import mean_squared_error

def write_pcd(filename, msg, mode='binary'):
    
    pc = PointCloud.from_msg(msg)
    pc.save_pcd(filename, compression=mode)
    
def read_pcd(filename):

    if not os.path.isfile(filename):
        raise Exception("[read_pcd] File does not exist.")
    pc = PointCloud.from_path(filename)

    return pc
    
def write_transformation(filename, transformation):
    np.savetxt(filename, transformation, delimiter=',',fmt='%.5f')

def write_img(filename, img):
    cv2.imwrite(filename, img)
    
def matrixToRodrigues(matrix):
    rods, _ = cv2.Rodrigues(matrix[0:3, 0:3])
    rods = rods.transpose()
    rodrigues = rods[0]
    return rodrigues

def matrixToQuaternion(matrix):
    rot_matrix = matrix[0:3, 0:3]
    r = R.from_matrix(rot_matrix)
    return r.as_quat()

def matrixToXYZ(matrix):
    return matrix[0:3,3]

def rodriguesToMatrix(r):
    rod = np.array(r, dtype=np.float)
    matrix = cv2.Rodrigues(rod)
    return matrix[0]

def quaternionToMatrix(quat):
    return R.from_quat(quat).as_matrix()

def poseToMatrix(pose):
    matrix = np.zeros((4,4))
    rot_mat = quaternionToMatrix(pose[3:])
    trans = pose[:3]
    matrix[0:3,0:3] = rot_mat
    matrix[0:3,3] = trans
    matrix[3,3] = 1
    return matrix

def write_intrinsic(filename, data):
    matrix = np.zeros((3,3))
    matrix[0,0] = data[0]
    matrix[0,1] = data[1]
    matrix[0,2] = data[2]
    matrix[1,0] = data[3]
    matrix[1,1] = data[4]
    matrix[1,2] = data[5]
    matrix[2,0] = data[6]
    matrix[2,1] = data[7]
    matrix[2,2] = data[8]
    
    np.savetxt(filename, matrix, delimiter=',',fmt='%.5f')

def rotationAndpositionToMatrix44(rotation, position):
    matrix44 = np.empty(shape=(4,4))
    matrix44[:3,:3] = rotation
    matrix44[:3,3] = position
    matrix44[3,:3] = 0
    matrix44[3,3] = 1
    
    return matrix44

    
def matrix44_to_pose(matrix44):
    quaternion = matrixToQuaternion(matrix44)
    quaternion = normalize_quat(quaternion)
    xyz = matrixToXYZ(matrix44)
    pose = np.append(xyz, quaternion) 
    return pose

def compute_position_error(pred, targ):
    pred = pred[:3]
    targ = targ[:3]
    
    return mean_squared_error(pred, targ, squared=False) # RMSE

def compute_rotation_error(pred, targ): 
     
    ## second way: using rodrigues (like ATOM) --> better because angle ranges from 0 to pi (whereas with quaterions ranges from 0 to 2pi)
    ## https://github.com/lardemua/atom/blob/284b7943e467e53a3258de6f673cf852b07654cb/atom_evaluation/scripts/camera_to_camera_evalutation.py#L290
    pred_matrix = poseToMatrix(pred)
    targ_matrix = poseToMatrix(targ)
    
    delta = np.dot(np.linalg.inv(pred_matrix), targ_matrix)
    deltaR = matrixToRodrigues(delta[0:3, 0:3])
    
    return np.linalg.norm(deltaR)

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
