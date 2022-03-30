import copy
import functools
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from colorama import Fore, Style
import cv2
import rospy
import tf

from geometry_msgs.msg import Pose

import localbot_core.src.pypcd as pypcd


def write_pcd(filename, msg, mode='binary'):
    
    pc = pypcd.PointCloud.from_msg(msg)
    pc.save_pcd(filename, compression=mode)
    
def read_pcd(filename):

    if not os.path.isfile(filename):
        raise Exception("[read_pcd] File does not exist.")
    pc = pypcd.PointCloud.from_path(filename)

    return pc
    
def write_transformation(filename, transformation):
    np.savetxt(filename, transformation, delimiter=',',fmt='%.5f')

def write_img(filename, img):
    cv2.imwrite(filename, img)
    
def data2pose(data):
    
    if type(data) is str:
        data = list(data)
        lst_data = [i for i in data if i!=','] # remove ','
        data = {'x'  : lst_data[0], 
                'y'  : lst_data[1], 
                'z'  : lst_data[2],
                'rx' : lst_data[3],
                'ry' : lst_data[4], 
                'rz' : lst_data[5]}
        
    quaternion = tf.transformations.quaternion_from_euler(data['rx'], data['ry'], data['rz'])
    p = Pose()
    p.position.x = data['x']
    p.position.y = data['y']
    p.position.z = data['z']
    
    p.orientation.x = quaternion[0]
    p.orientation.y = quaternion[1]
    p.orientation.z = quaternion[2]
    p.orientation.w = quaternion[3]
        
    return p

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
    
    
    

