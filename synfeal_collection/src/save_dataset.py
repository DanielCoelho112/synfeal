#!/usr/bin/env python3

import glob
import rospy
import os
from torchvision import transforms
from PIL import Image as PILImage
from visualization_msgs.msg import *
from cv_bridge import CvBridge
from tf.listener import TransformListener
from utils import write_intrinsic, write_img, write_transformation
from utils_ros import read_pcd, write_pcd
from sensor_msgs.msg import PointCloud2, Image, PointField, CameraInfo
from colorama import Fore , Style
from datetime import datetime
import yaml
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class SaveDataset():
    def __init__(self, output, mode, dbf = None, uvl = None, model3d_config = None, fast=False):
        
        path=os.environ.get("SYNFEAL_DATASET")
        self.output_folder = f'{path}/datasets/localbot/{output}'

        ans = ''
        self.continue_dataset = False
        if os.path.exists(self.output_folder):
            print(Fore.YELLOW + f'Dataset already exists! Do you want to continue?' + Style.RESET_ALL)
            ans = input(Fore.YELLOW + "Y" + Style.RESET_ALL + "es/" + Fore.YELLOW + "N" + Style.RESET_ALL + "o/" + Fore.YELLOW +'O' + Style.RESET_ALL + 'verwrite: ') # Asks the user if they want to resume training
            
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        elif os.path.exists(self.output_folder) and ans.lower() in ['o' , 'overwrite']:
            print(f'Overwriting folder {self.output_folder}')
            os.system(f'rm -r {self.output_folder}')
            os.makedirs(self.output_folder) 
        elif os.path.exists(self.output_folder) and ans.lower() in ['' , 'y' , 'yes']:
            print(f'Continuing with folder {self.output_folder}')
            images = glob.glob(f'{self.output_folder}/*.rgb.png')
            self.continue_dataset = True
            frame_idx = len(images)
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveDataset initialization! {Fore.RESET}')
            exit(0)

        self.rgb_transform = transforms.Compose([
            transforms.Resize(400),
        ])
        
        self.resize_image = model3d_config['resize_image'] if model3d_config is not None else False
        name_model3d_config = model3d_config if model3d_config is not None else None 
         
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
                  'model3d_config' : name_model3d_config,
                  'fast' : fast}
        
        self.fast = fast
        if self.continue_dataset:
            self.frame_idx = frame_idx 
        else:
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
        
        with open(f'{self.output_folder}/model3d_config.yaml', 'w') as file:
            yaml.dump(model3d_config, file)
        
        
        print('SaveDataset initialized properly')

        
    def saveFrame(self):
        
        transformation = self.getTransformation()
        image = self.getImage()

        if self.resize_image:
            image = self.rgb_transform(PILImage.fromarray(image))
            image = np.array(image)
        
        
        filename = f'frame-{self.frame_idx:05d}'
                
        write_transformation(f'{self.output_folder}/{filename}.pose.txt', transformation)    
        write_img(f'{self.output_folder}/{filename}.rgb.png', image)
        
        if not self.fast:
            pc_msg = self.getPointCloud()
            write_pcd(f'{self.output_folder}/{filename}.pcd', pc_msg)
        
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
            
        
        

    
        
        