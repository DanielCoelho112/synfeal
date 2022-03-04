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

def write_pcd(filename, msg, mode='binary'):
    
    pc = pypcd.PointCloud.from_msg(msg)
    pc.save_pcd(filename, compression=mode)
    
def write_trans_rot(filename, trans, rot):
    with open(filename, 'w') as f:
        f.write(f't {trans[0]} {trans[1]} {trans[2]}')
        f.write('\n')
        f.write(f'r {rot[0]} {rot[1]} {rot[2]} {rot[3]}')

def write_img(filename, img):
    cv2.imwrite(filename, img)
    
