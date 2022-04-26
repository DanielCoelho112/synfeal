#!/usr/bin/env python3

# stdlib

import sys
import argparse
import copy
import random
import math



# 3rd-party
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, SetModelStateRequest
import tf
import numpy as np
from localbot_core.src.utilities import *
from colorama import Fore
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GenerateRealPredicted():
    
    def __init__(self, model_name, results):
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
        self.menu_handler = MenuHandler()
        self.model_name = model_name # model_name = 'localbot'
        self.bridge = CvBridge()
        self.folder = f'{results.path}/images'
        
        if not os.path.exists(self.folder):
            print(f'Creating folder {self.folder}')
            os.makedirs(self.folder)  # Create the new folder
        else:
            print(f'{Fore.RED} {self.folder} already exists... Aborting GenerateRealPredicted initialization! {Fore.RESET}')
            exit(0)
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
    def getPose(self):
        return self.get_model_state_service(self.model_name, 'world')
    
    def setPose(self, pose):
        
        req = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        req.model_state.model_name = self.model_name
        req.model_state.pose.position.x = pose.position.x
        req.model_state.pose.position.y = pose.position.y
        req.model_state.pose.position.z = pose.position.z
        req.model_state.pose.orientation.x = pose.orientation.x
        req.model_state.pose.orientation.y = pose.orientation.y
        req.model_state.pose.orientation.z = pose.orientation.z
        req.model_state.pose.orientation.w = pose.orientation.w
        req.model_state.reference_frame = 'world'
        self.set_state_service(req.model_state)
    
    def getImage(self):
        rgb_msg = rospy.wait_for_message('/kinect/rgb/image_raw', Image)
        return self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8") # convert to opencv image
        
    def saveImage(self, filename, image):
        filename = f'{self.folder}/{filename}'
        write_img(filename, image)
        