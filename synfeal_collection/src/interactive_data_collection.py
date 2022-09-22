#!/usr/bin/env python3

import copy
import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from gazebo_msgs.srv import SetModelState, GetModelState, SetModelStateRequest
from synfeal_collection.src.save_dataset import SaveDataset


class InteractiveDataCollection():
    
    def __init__(self, model_name, seq):
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) 
        self.menu_handler = MenuHandler()
        self.model_name = model_name
        self.server = InteractiveMarkerServer("interactive_camera")
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        pose_gazebo = self.get_model_state_service(self.model_name, 'world')
        self.pose = copy.deepcopy(pose_gazebo.pose)
        self.make6DofMarker(True, InteractiveMarkerControl.MOVE_3D, pose_gazebo.pose, True)
        
        # add interactive marker to save datasets
        self.original_pose = Pose(position=Point(x=0, y=0, z=1), orientation=Quaternion(x=0, y=0, z=0, w=1))
        self.makeClickMarker(self.original_pose)

        self.server.applyChanges()
        
        # create instance to save dataset
        self.save_dataset = SaveDataset(seq, mode='interactive')
        
    
    def makeBox(self, msg, pose, color):
        marker = Marker(header=Header(frame_id="world", stamp=rospy.Time.now()),
                   ns=self.model_name, id=0, frame_locked=False,
                   type=Marker.SPHERE, action=Marker.ADD, lifetime=rospy.Duration(0),
                   pose=pose,
                   scale=Vector3(x=0.05, y=0.05, z=0.05),
                   color=ColorRGBA(r=color[0], g=color[1], b=color[2], a=1))
        return marker
        
    def makeBoxControl(self, msg, pose, color):
        control =  InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.always_visible = True
        control.markers.append(self.makeBox(msg, pose, color))
        msg.controls.append(control)
        return control

    def makeClickMarker(self,pose):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.pose = pose
        int_marker.scale = 1
        int_marker.name = "Save Frame"
        int_marker.description = "Click to save frame"
        control = self.makeBoxControl(int_marker, pose, color= [0.2,0.8,0.2])
        int_marker.controls.append(copy.deepcopy(control))
            
        self.server.insert(int_marker, self.processFeedbackMenu)
        self.menu_handler.apply(self.server, int_marker.name)

    def make6DofMarker(self, fixed, interaction_mode, pose, show_6dof=False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.pose = pose
        int_marker.scale = 0.3

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        self.makeBoxControl(int_marker, pose, color= [0.8,0.2,0.2])
        int_marker.controls[0].interaction_mode = interaction_mode

        if fixed:
            int_marker.name += "_fixed"
            int_marker.description += "\n(fixed orientation)"

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D"}
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof:
                int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]
                
        if show_6dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

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
            
    def callbackTimer(self,event):
        print('Timer called at ' + str(event.current_real))
        
    def getFrameIdx(self):
        return self.save_dataset.frame_idx
        
