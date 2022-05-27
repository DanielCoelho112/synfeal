import copy
import functools
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from colorama import Fore, Style
import cv2
#import tf
#from geometry_msgs.msg import Pose
from localbot_core.src.pypcd_no_ros import PointCloud
from localbot_localization.src.utilities import *

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
    