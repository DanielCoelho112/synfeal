#!/usr/bin/env python3

# 3rd-party

import rospy
import os
from visualization_msgs.msg import *
from cv_bridge import CvBridge
from tf.listener import TransformListener
from localbot_core.src.utilities import *
from sensor_msgs.msg import PointCloud2, Image, PointField
from colorama import Fore
from datetime import datetime
import yaml

class SaveDataset():
    """
    class to save datasets
    once initialized, we can call the method 'saveFrame' to save to disk the image, point cloud w.r.t frame frame and rgb_frame transformation.
    """
    
    def __init__(self, output, mode):
        
        # attribute initializer
        self.output_folder = f'{os.environ["HOME"]}/datasets/localbot/{output}'
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveDataset initialization! {Fore.RESET}')
            exit(0)
        
        
        dt_now = datetime.now() # current date and time
        config = {'user' : os.environ["USER"],
                  'date' : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'mode' : mode}
        
        with open(f'{self.output_folder}/log.yaml', 'w') as file:
            yaml.dump(config, file)
        
        
        self.frame_idx = 0 # make sure to save as 00000
        self.world_link = 'world'
        self.depth_frame = 'kinect_depth_optical_frame'
        self.rgb_frame = 'kinect_rgb_optical_frame'
        
        
        self.trans = None # 1x3 translation
        self.rot = None # 1x4 quaternion
        self.image = None # rgb image
        self.pc_msg = None # point cloud w.r.t rgb_frame
        
        self.listener = TransformListener()
        self.bridge = CvBridge()
        
        # get transformation from depth_frame to rgb_fram
        now = rospy.Time()
        print(f'Waiting for transformation from {self.depth_frame} to {self.rgb_frame}')
        self.listener.waitForTransform(self.depth_frame, self.rgb_frame , now, rospy.Duration(5)) # admissible waiting time
        print('... received!')
        self.transform_depth_rgb = self.listener.lookupTransform(self.depth_frame, self.rgb_frame, now)
        self.matrix_depth_rgb = self.listener.fromTranslationRotation(self.transform_depth_rgb[0], self.transform_depth_rgb[1])
        
        print('SaveDataset initialized properly')

        
    def saveFrame(self):
        
        print('getting trasformation...')
        self.getTransformation()
        
        print('getting image...')
        self.getImage()
        
        print('getting pointcloud')
        self.getPointCloud()
        
        print('saving files to disk...')
        filename = f'frame-{self.frame_idx:05d}'
        write_trans_rot(f'{self.output_folder}/{filename}.pose.txt', self.trans, self.rot)
        write_pcd(f'{self.output_folder}/{filename}.pcd', self.pc_msg)
        write_img(f'{self.output_folder}/{filename}.rgb.png', self.image)
        
        print(f'frame-{self.frame_idx:05d} saved successfully')
        
        self.frame_idx+=1
        
        
    def getTransformation(self):
        
        now = rospy.Time()
        print(f'Waiting for transformation from {self.world_link} to {self.rgb_frame}')
        self.listener.waitForTransform(self.world_link, self.rgb_frame , now, rospy.Duration(5))
        print('... received!')
        (trans,rot) = self.listener.lookupTransform(self.world_link, self.rgb_frame, now)
        self.trans = trans
        self.rot = rot
        
    def getImage(self):
        
        rgb_msg = rospy.wait_for_message('/kinect/rgb/image_raw', Image)
        self.image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8") # convert to opencv image

    def getPointCloud(self):
        
        pc_msg = rospy.wait_for_message('/kinect/depth/points', PointCloud2)
        pc2_points = pc2.read_points(pc_msg)
        gen_selected_points = list(pc2_points)
        lst_points = []
        for point in gen_selected_points:
            lst_points.append([point[0], point[1], point[2], 1])
            
        np_points = np.array(lst_points)
        
        # convert to rgb_frame
        np_points = np.dot(np_points, self.matrix_depth_rgb)
        
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]
        
        pc_msg.header.frame_id = self.rgb_frame
        self.pc_msg = pc2.create_cloud(pc_msg.header, fields, np_points)
            
        
        
        

    
        
        