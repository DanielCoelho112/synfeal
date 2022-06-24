#!/usr/bin/env python3

import rospy
import os
from visualization_msgs.msg import *
from cv_bridge import CvBridge
from tf.listener import TransformListener
from localbot_core.src.utilities_ros import *
from sensor_msgs.msg import PointCloud2, Image, PointField, CameraInfo
from colorama import Fore
from datetime import datetime
import yaml
import sensor_msgs.point_cloud2 as pc2

class SaveDataset():
    """
    class to save datasets
    once initialized, we can call the method 'saveFrame' to save to disk the image, point cloud w.r.t frame frame and rgb_frame transformation.
    """    
    def __init__(self, output, mode, dbf = None, uvl = None, model3d_config_name = None):
        
        self.output_folder = f'{os.environ["HOME"]}/datasets/localbot/{output}'
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveDataset initialization! {Fore.RESET}')
            exit(0)
        
        dt_now = datetime.now() # current date and time
        config = {'user'     : os.environ["USER"],
                  'date'     : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'mode'     : mode,
                  'is_valid' : False,
                  'npoints'  : None,
                  'scaled'   : False,
                  'distance_between_frames'  : dbf,
                  'raw'      : output,
                  'variable_lights' : uvl,
                  'model3d_config' : model3d_config_name}
        
        self.frame_idx = 0 
        self.world_link = 'world'
        self.depth_frame = 'kinect_depth_optical_frame'
        self.rgb_frame = 'kinect_rgb_optical_frame'
        
        self.listener = TransformListener()
        self.bridge = CvBridge()
        
        # get transformation from depth_frame to rgb_fram
        now = rospy.Time()
        print(f'Waiting for transformation from {self.depth_frame} to {self.rgb_frame}')
        self.listener.waitForTransform(self.depth_frame, self.rgb_frame , now, rospy.Duration(5)) # admissible waiting time
        print('... received!')
        self.transform_depth_rgb = self.listener.lookupTransform(self.depth_frame, self.rgb_frame, now)
        self.matrix_depth_rgb = self.listener.fromTranslationRotation(self.transform_depth_rgb[0], self.transform_depth_rgb[1])
        
        # get intrinsic matrices from both cameras
        rgb_camera_info = rospy.wait_for_message('/kinect/rgb/camera_info', CameraInfo)
        depth_camera_info = rospy.wait_for_message('/kinect/depth/camera_info', CameraInfo)
        
        # rgb information
        rgb_intrinsic = rgb_camera_info.K
        rgb_width = rgb_camera_info.width
        rgb_height = rgb_camera_info.height
        
        # depth information
        depth_width = depth_camera_info.width
        depth_height = depth_camera_info.height
        depth_intrinsic = depth_camera_info.K
        
        # save intrinsic to txt file
        write_intrinsic(f'{self.output_folder}/rgb_intrinsic.txt', rgb_intrinsic)
        write_intrinsic(f'{self.output_folder}/depth_intrinsic.txt', depth_intrinsic)
        
        rgb_dict = {'intrinsic' : f'{self.output_folder}/rgb_intrinsic.txt',
                    'width'     : rgb_width,
                    'height'    : rgb_height}
        
        depth_dict = {'intrinsic' : f'{self.output_folder}/depth_intrinsic.txt',
                    'width'       : depth_width,
                    'height'      : depth_height}
        
        config['rgb'] = rgb_dict
        config['depth'] = depth_dict
        
        with open(f'{self.output_folder}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        
        print('SaveDataset initialized properly')

        
    def saveFrame(self):
        
        transformation = self.getTransformation()
        image = self.getImage()
        pc_msg = self.getPointCloud()
        
        filename = f'frame-{self.frame_idx:05d}'
        write_transformation(f'{self.output_folder}/{filename}.pose.txt', transformation)
        write_pcd(f'{self.output_folder}/{filename}.pcd', pc_msg)
        write_img(f'{self.output_folder}/{filename}.rgb.png', image)
        
        print(f'frame-{self.frame_idx:05d} saved successfully')
        
        self.step()
                
    def getTransformation(self):
        
        now = rospy.Time()
        print(f'Waiting for transformation from {self.world_link} to {self.rgb_frame}')
        self.listener.waitForTransform(self.world_link, self.rgb_frame , now, rospy.Duration(5))
        print('... received!')
        (trans,rot) = self.listener.lookupTransform(self.world_link, self.rgb_frame, now)
        return self.listener.fromTranslationRotation(trans, rot)
        
    def getImage(self):
        
        rgb_msg = rospy.wait_for_message('/kinect/rgb/image_raw', Image)
        return self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8") # convert to opencv image

    def getPointCloud(self):
        
        pc_msg = rospy.wait_for_message('/kinect/depth/points', PointCloud2)
        pc2_points = pc2.read_points(pc_msg)
        gen_selected_points = list(pc2_points)
        lst_points = []
        for point in gen_selected_points:
            lst_points.append([point[0], point[1], point[2], 1])
            
        np_points = np.array(lst_points)
        
        # convert to rgb_frame
        np_points = np.dot(self.matrix_depth_rgb, np_points.T).T
        
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]
        
        pc_msg.header.frame_id = self.rgb_frame
        return pc2.create_cloud(pc_msg.header, fields, np_points)
    
    def step(self):
        self.frame_idx+=1
            
        
#SaveDataset('q', 'automatic')
        

    
        
        