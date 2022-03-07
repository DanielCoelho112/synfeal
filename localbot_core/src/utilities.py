import copy
# stdlib
import functools
import json
# 3rd-party
import numpy as np
import os

# 3rd-party
# import pypcd
from colorama import Fore, Style

import cv2
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from sensor_msgs.msg import PointCloud2
import imageio
import atom_core.pypcd as pypcd
from geometry_msgs.msg import Point, Pose, Quaternion

def write_pcd(filename, msg, mode='binary'):
    
    pc = pypcd.PointCloud.from_msg(msg)
    pc.save_pcd(filename, compression=mode)
    
def write_transformation(filename, transformation):
    with open(filename, 'w') as f:
        f.write(str(transformation))

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
