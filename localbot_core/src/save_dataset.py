#!/usr/bin/env python3

# stdlib
import sys
import argparse
import copy

# 3rd-party
import rospkg
import rospy
from colorama import Fore, Style
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from atom_core.ros_utils import filterLaunchArguments
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, SetModelStateRequest
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from tf.listener import TransformListener
from localbot_core.src.utilities import *
from sensor_msgs.msg import PointCloud2, Image, PointField


import warnings
warnings.filterwarnings("ignore")
# not needed here I guess
#rospy.init_node("interactive_camera")

class SaveDataset():
    # save : point cloud; image; transformation; 
    
    # this is responsable to keep track of the idx
    
    def __init__(self, output=None):
        
        self.output_folder = '/home/danc/datasets/localbot/test'
        self.listener = TransformListener()
        self.bridge = CvBridge()
        self.world_link = 'world'
        self.target_link = 'kinect_depth_optical_frame'
        
        self.trans = None
        self.rot = None
        self.image = None
        self.pc_msg = None
        
        self.depth_frame = 'kinect_depth_optical_frame'
        self.rgb_frame = 'kinect_rgb_optical_frame'
        
        now = rospy.Time()
        
        print(f'Waiting for transformation from {self.depth_frame} to {self.rgb_frame}')
        self.listener.waitForTransform(self.depth_frame, self.rgb_frame , now, rospy.Duration(5)) # wait until it becomes available
        print('... received!')
        self.transforms_depth_rgb = self.listener.lookupTransform(self.depth_frame, self.rgb_frame, now)
        self.matrix_depth_rgb = self.listener.fromTranslationRotation(self.transforms_depth_rgb[0], self.transforms_depth_rgb[1])
        
        print(self.transforms_depth_rgb[0])
        print(self.matrix_depth_rgb)
        
        
        
        
        
    def saveFrame(self):
        self.getTransformation()
        print('begin image')
        self.getImage()
        print('begin point cloud')
        self.getPointCloud()
        print('end poit cloud')
        
        # save files
        write_trans_rot(self.output_folder + '/trans_rot.txt', self.trans, self.rot)
        write_pcd(self.output_folder + '/poo.pcd', self.pc_msg)
        write_img(self.output_folder + '/image.png', self.image)
        
        print('frame saved')
        
        
    def getTransformation(self):
        now = rospy.Time()
        
        print(f'Waiting for transformation from {self.world_link} to {self.target_link}')
        self.listener.waitForTransform(self.world_link, self.target_link , now, rospy.Duration(5)) # wait until it becomes available
        print('... received!')
        (trans,rot) = self.listener.lookupTransform(self.world_link, self.target_link, now)
        self.trans = trans
        self.rot = rot
        
        
        
    def getImage(self):
        msg = rospy.wait_for_message('/kinect/rgb/image_raw', Image)
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to opencv image and save image to disk
        #     filename = self.output_folder + '/' + sensor['_name'] + '_' + str(self.data_stamp) + '.jpg'
        #     filename_relative = sensor['_name'] + '_' + str(self.data_stamp) + '.jpg'
        #     cv2.imwrite(filename, cv_image)
        
    def getPointCloud(self):
        pc_msg = rospy.wait_for_message('/kinect/depth/points', PointCloud2)
    
        points_ = pc2.read_points(pc_msg)
        gen_selected_points = list(points_)
        points_ = []
        print('1')
        for point in gen_selected_points:
            points_.append([point[0], point[1], point[2], 1])
            
        print('2')
        
        points_np = np.array(points_)
        print(points_np.shape)
        
        print(self.matrix_depth_rgb.shape)
        final_points_np = np.dot(points_np, self.matrix_depth_rgb)
        
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]

        
        pc_msg.header.frame_id = self.rgb_frame
        self.pc_msg = pc2.create_cloud(pc_msg.header, fields, final_points_np)
            
        
        
        

    
        
        