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
from localbot_core.src.save_dataset import SaveDataset
import tf



class AutomaticDataCollection():
    
    def __init__(self, model_name, seq):
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
        self.menu_handler = MenuHandler()
        self.model_name = model_name # model_name = 'localbot'
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        pose_gazebo = self.get_model_state_service(self.model_name, 'world')
        
        #self.pose = copy.deepcopy(pose_gazebo.pose)

        # create instance to save dataset
        self.save_dataset = SaveDataset(f'seq{seq}', mode='automatic')
        
        # define minimum and maximum boundaries
        self.x_min = 1
        self.x_max = 4
        self.y_min = 1
        self.y_max = 5
        self.z_min = -1
        self.z_max = 1
        
        
    def generateRandomPose(self):
        
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        
        rx = random.uniform(0, math.pi/4)
        ry = random.uniform(0, math.pi/4)
        rz = random.uniform(0, math.pi/4)
        
        quaternion = tf.transformations.quaternion_from_euler(rx, ry, rz)
        
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        # Make sure the quaternion is valid and normalized
        
        p.orientation.x = quaternion[0]
        p.orientation.y = quaternion[1]
        p.orientation.z = quaternion[2]
        p.orientation.w = quaternion[3]
        
        return p
            
    def generatePath(self):
        pass
    
    def getPose(self):
        pass
    
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
        # self.server.applyChanges()    # needed???

    def saveFrame(self):
        self.save_dataset.saveFrame()


            

        
