import copy
import functools
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from colorama import Fore, Style
import cv2
import tf
from geometry_msgs.msg import Pose
from localbot_core.src.pypcd import PointCloud


    
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
    #quaternion = R.from_euler('xyz',[[data['rx'], data['ry'], data['rz']]], degrees=False).as_quat()
    
    p = Pose()
    p.position.x = data['x']
    p.position.y = data['y']
    p.position.z = data['z']
    
    p.orientation.x = quaternion[0]
    p.orientation.y = quaternion[1]
    p.orientation.z = quaternion[2]
    p.orientation.w = quaternion[3]
        
    return p
