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
import numpy as np
from localbot_core.src.utilities import *
from localbot_core.src.utilities_ros import *
from colorama import Fore



class AutomaticDataCollection():
    
    def __init__(self, model_name, seq, dbf = None, usv = None):
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
        self.menu_handler = MenuHandler()
        self.model_name = model_name # model_name = 'localbot'
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # create instance to save dataset
        self.save_dataset = SaveDataset(f'{seq}', mode='automatic', dbf = dbf, usv = usv)
        
        # define minimum and maximum boundaries
        self.x_min = 1.5
        self.x_max = 3.5
        self.y_min = 1.5
        self.y_max = 5.0
        self.z_min = -0.5
        self.z_max = 0.5
        
        # define minimum and maximum light
        self.att_min = 0.1
        self.att_max = 1.0
        
        self.light_names = ['user_point_light_0', 'user_point_light_1', 'user_point_light_2', 'user_point_light_3', 'user_point_light_4', 'user_point_light_5']
        
        self.att_initial = 1.0
        
        
    def generateRandomPose(self):
        
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)
        
        rx = random.uniform(-math.pi/6, math.pi/6)
        ry = random.uniform(-math.pi/6, math.pi/6)
        rz = random.uniform(0, 2*math.pi)
        
        quaternion = tf.transformations.quaternion_from_euler(rx, ry, rz)
        
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        
        p.orientation.x = quaternion[0]
        p.orientation.y = quaternion[1]
        p.orientation.z = quaternion[2]
        p.orientation.w = quaternion[3]
        
        return p
            
    def generatePath(self, dbf,  final_pose = None):
        
        initial_pose = self.getPose().pose
        
        if final_pose == None:
            final_pose = self.generateRandomPose()
        
            while True:
                xyz_initial = np.array([initial_pose.position.x, initial_pose.position.y, initial_pose.position.z])
                xyz_final = np.array([final_pose.position.x, final_pose.position.y, final_pose.position.z])
                l2_dst = np.linalg.norm(xyz_final - xyz_initial)
                
                # if final pose is close to the initial choose another final pose
                if l2_dst < 1.5:
                    final_pose = self.generateRandomPose()
                else:
                    break
                
                
        # compute n_steps based on l2_dist
        n_steps = int(l2_dst / dbf)
        
        print('using n_steps of: ', n_steps)
        
        
        step_poses = [] # list of tuples
        rx, ry, rz = tf.transformations.euler_from_quaternion([initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w])
        pose_initial_dct = {'x'  : initial_pose.position.x, 
                            'y'  : initial_pose.position.y, 
                            'z'  : initial_pose.position.z, 
                            'rx' : rx,
                            'ry' : ry, 
                            'rz' : rz}
        
        rx, ry, rz = tf.transformations.euler_from_quaternion([final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w])
        pose_final_dct =  {'x'  : final_pose.position.x, 
                           'y'  : final_pose.position.y, 
                           'z'  : final_pose.position.z, 
                           'rx' : rx,
                           'ry' : ry, 
                           'rz' : rz}
            
        x_step_var = (pose_final_dct['x'] - pose_initial_dct['x']) / n_steps
        y_step_var = (pose_final_dct['y'] - pose_initial_dct['y']) / n_steps
        z_step_var = (pose_final_dct['z'] - pose_initial_dct['z']) / n_steps
        rx_step_var = (pose_final_dct['rx'] - pose_initial_dct['rx']) / n_steps
        ry_step_var = (pose_final_dct['ry'] - pose_initial_dct['ry']) / n_steps
        rz_step_var = (pose_final_dct['rz'] - pose_initial_dct['rz']) / n_steps
        
        for i in range(n_steps):
            dct = {'x'  : pose_initial_dct['x'] + (i + 1) * x_step_var, 
                   'y'  : pose_initial_dct['y'] + (i + 1) * y_step_var, 
                   'z'  : pose_initial_dct['z'] + (i + 1) * z_step_var,
                   'rx' : pose_initial_dct['rx'] + (i + 1) * rx_step_var,
                   'ry' : pose_initial_dct['ry'] + (i + 1) * ry_step_var, 
                   'rz' : pose_initial_dct['rz'] + (i + 1) * rz_step_var}
            pose = data2pose(dct)
            step_poses.append(pose)
        
        return step_poses
            

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
    
    def generateLights(self, n_steps, random):
        lights = []
        if random:
            lights = [np.random.uniform(low=self.att_min, high=self.att_max) for _ in range(n_steps)]
        else:
            initial_light = self.att_initial
            final_light = np.random.uniform(low=self.att_min, high=self.att_max)
            
            step_light = (final_light - initial_light) / n_steps
            
            for i in range(n_steps):
                lights.append(initial_light + (i + 1) * step_light)
        
            self.att_initial = final_light
        return lights
                
    def setLight(self, light):
        
        for name in self.light_names:
            
            my_str = f'name: "{name}" \nattenuation_quadratic: {light}'
    
            with open('/tmp/set_light.txt', 'w') as f:
                f.write(my_str)
            
            os.system(f'gz topic -p /gazebo/my_room_024/light/modify -f /tmp/set_light.txt')
                    
        
    def saveFrame(self):
        self.save_dataset.saveFrame()
        
    def getFrameIdx(self):
        return self.save_dataset.frame_idx


            

        
