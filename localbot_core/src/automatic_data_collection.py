#!/usr/bin/env python3

# stdlib
import sys
import argparse
import copy

# 3rd-party
import rospy
from colorama import Fore, Style
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, SetModelStateRequest
from localbot_core.src.save_dataset import SaveDataset


class AutomaticDataCollection():
    
    def __init__(self, model_name):
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
        self.menu_handler = MenuHandler()
        self.model_name = model_name # model_name = 'localbot'
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        pose_gazebo = self.get_model_state_service(self.model_name, 'world')
        
        self.pose = copy.deepcopy(pose_gazebo.pose)

        # create instance to save dataset
        self.save_dataset = SaveDataset('seq05', mode='interactive')
        
    def generateRandomPose(self):
        pass
        # x, y, z, rx, ry, rz
        # x,y,z with constrains
        # convert rx, ry, rz to quaternion
    
    def generatePath(self):
        pass
    
    def getPose(self):
        pass
    
    def setPose(self):
        pass

    def saveFrame(self):
        pass
        
        

    def processFeedback(self, feedback):
        s = "feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"

        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo( s + ": pose changed")
            print('feedback = \n' + str(feedback))
            
            self.pose.position.x = feedback.pose.position.x
            self.pose.position.y = feedback.pose.position.y
            self.pose.position.z = feedback.pose.position.z
            self.pose.orientation.x = feedback.pose.orientation.x
            self.pose.orientation.y = feedback.pose.orientation.y
            self.pose.orientation.z = feedback.pose.orientation.z
            self.pose.orientation.w = feedback.pose.orientation.w

            req = SetModelStateRequest()  # Create an object of type SetModelStateRequest

            req.model_state.model_name = self.model_name
            req.model_state.pose.position.x = self.pose.position.x
            req.model_state.pose.position.y = self.pose.position.y
            req.model_state.pose.position.z = self.pose.position.z
            req.model_state.pose.orientation.x = self.pose.orientation.x
            req.model_state.pose.orientation.y = self.pose.orientation.y
            req.model_state.pose.orientation.z = self.pose.orientation.z
            req.model_state.pose.orientation.w = self.pose.orientation.w
            req.model_state.reference_frame = 'world'

            self.set_state_service(req.model_state)
            self.server.applyChanges()

    def processFeedbackMenu(self, feedback):
        s = "feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"
    
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.save_dataset.saveFrame()
            

        
