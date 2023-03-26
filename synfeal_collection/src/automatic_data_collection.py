#!/usr/bin/env python3

# stdlib

import random
import os
from xml.parsers.expat import model

import pandas as pd
# 3rd-party
import rospy
import tf
import numpy as np
import trimesh
from geometry_msgs.msg import Pose , Quaternion , Point , Vector3
#from interactive_markers.interactive_marker_server import *
#from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from gazebo_msgs.srv import SetModelState, GetModelState, SetModelStateRequest
from gazebo_msgs.srv import SetLightProperties , SetLightPropertiesRequest , SpawnModel , SpawnModelRequest , DeleteModel , DeleteModelRequest
from colorama import Fore, Style
from scipy.spatial.transform import Rotation as R
import pvlib
from pvlib.location import Location
import datetime

from synfeal_collection.src.save_dataset import SaveDataset
from utils import *
from utils_ros import *


class AutomaticDataCollection():
    
    def __init__(self, model_name, seq, dbf=None, uvl=None, use_objects = None , model3d_config=None, fast=None, save_dataset=True, mode=None):
        self.model_name = model_name  # model_name = 'localbot'
        self.dbf = dbf
    
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        rospy.wait_for_service('/gazebo/set_light_properties')
        self.modify_light = rospy.ServiceProxy('/gazebo/set_light_properties', SetLightProperties)

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        rospy.wait_for_service('/gazebo/delete_model')
        self.delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # create instance to save dataset
        if save_dataset:
            self.save_dataset = SaveDataset(
                f'{seq}', mode=mode, dbf=dbf, uvl=uvl, model3d_config=model3d_config, fast=fast)

        name_model3d_config = model3d_config['name'].split('.')[0]
        # define minimum and maximum boundaries
        self.x_min = model3d_config['volume']['position']['xmin']
        self.x_max = model3d_config['volume']['position']['xmax']
        self.y_min = model3d_config['volume']['position']['ymin']
        self.y_max = model3d_config['volume']['position']['ymax']
        self.z_min = model3d_config['volume']['position']['zmin']
        self.z_max = model3d_config['volume']['position']['zmax']

        self.rx_min = model3d_config['volume']['angles']['rxmin']
        self.rx_max = model3d_config['volume']['angles']['rxmax']
        self.ry_min = model3d_config['volume']['angles']['rymin']
        self.ry_max = model3d_config['volume']['angles']['rymax']
        self.rz_min = model3d_config['volume']['angles']['rzmin']
        self.rz_max = model3d_config['volume']['angles']['rzmax']

        # define minimum and maximum light
        self.att_min = model3d_config['light']['att_min']
        self.att_max = model3d_config['light']['att_max']
        self.att_initial = model3d_config['light']['att_initial']

        self.lights = model3d_config['lights']

        self.objects = model3d_config['objects']

        self.roll_initial = model3d_config['sun']['roll_initial']
        self.pitch_initial = model3d_config['sun']['pitch_initial']
        self.yaw_initial = model3d_config['sun']['yaw_initial']
        self.initial_time = datetime.datetime(2021, 6, 1, 0, 0, 0)
        self.site = Location(40.456, -3.73, 'Etc/GMT+1', 651, 'Ciemat (Madrid, ES)') # latitude, longitude, time_zone, altitude, name

        self.use_collision = model3d_config['collision']['use']
        self.min_cam_dist = model3d_config['collision']['min_camera_distance']

        self.path=os.environ.get("SYNFEAL_DATASET")

        # Load meshes
        if self.use_collision:
            self.mesh_collision = trimesh.load(
                f'{self.path}/models_3d/localbot/{name_model3d_config}/{name_model3d_config}_collision.dae', force='mesh')
        else:
            self.mesh_collision = False

        if use_objects:
            # Stores the resting pose for the objects when not in use
            self.resting_pose = Pose()
            self.resting_pose.position.x = model3d_config['objects_resting_pose']['x']
            self.resting_pose.position.y = model3d_config['objects_resting_pose']['y']
            self.resting_pose.position.z = model3d_config['objects_resting_pose']['z']
            # Loads the mesh of the model
            for object in self.objects:
                object['mesh'] = trimesh.load(f'{self.path}/models_3d/localbot/Objects/{object["name"]}/meshes/{object["mesh_name"]}', force='mesh')

        # set initial pose
        print('setting initial pose...')
        x = model3d_config['initial_pose']['x']
        y = model3d_config['initial_pose']['y']
        z = model3d_config['initial_pose']['z']
        rx = model3d_config['initial_pose']['rx']
        ry = model3d_config['initial_pose']['ry']
        rz = model3d_config['initial_pose']['rz']
        quaternion = tf.transformations.quaternion_from_euler(rx, ry, rz)
        p = Pose()
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation.x = quaternion[0]
        p.orientation.y = quaternion[1]
        p.orientation.z = quaternion[2]
        p.orientation.w = quaternion[3]
        self.setPose(model_name , p)
        rospy.sleep(1)

    def generateRandomPose(self):

        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        z = random.uniform(self.z_min, self.z_max)

        rx = random.uniform(self.rx_min, self.rx_max)
        ry = random.uniform(self.ry_min, self.ry_max)
        rz = random.uniform(self.rz_min, self.rz_max)

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
    
    def generateRandomPoseInsideMesh(self , camera_poses = [Pose()], manager = []):
        final_poses = []
        object_names = []
        # Checks if there are more than 10 objects available if so only choose 10
        if len(self.objects)>10:
            objects_sampled = random.sample(self.objects,10)
        else:
            objects_sampled = self.objects

        for object in objects_sampled:
            print(f'Generating pose for {object["name"]}')
            while True:
                p = self.generateRandomPose()
                translation = np.array([p.position.x, p.position.y, p.position.z])
                object_mesh = object['mesh'].copy()
                # TODO perform this transformation in the convex hull
                rotation_matrix = trimesh.transformations.quaternion_matrix([0, 0, p.orientation.z, p.orientation.w])
                object_mesh.apply_transform(rotation_matrix)
                object_mesh.apply_translation(translation)
                points = trimesh.convex.hull_points(object_mesh)
                del object_mesh # this variable lead to a memory leak
                is_inside = self.checkInsideMesh(points) # Check if the object is inside the mesh
                # If the object is not inside the mesh, try again
                if not is_inside:
                    continue
                # Check if the object collides with the camera
                valid = True
                for camera_pose in camera_poses:
                    if abs(p.position.x - camera_pose.position.x) < 1 and abs(p.position.y - camera_pose.position.y) < 1:
                        valid = False
                        print(f'{Fore.RED}Collision with camera{Style.RESET_ALL} Generating new pose...')
                        break
                if valid == False:
                    continue
                # Check if the object collides with previous objects
                valid = True
                for pose in final_poses:
                    if abs(pose.position.x - p.position.x) < 1 and abs(pose.position.y - p.position.y) < 1:
                        valid = False
                        print(f'{Fore.RED}Collision with another object{Style.RESET_ALL} Generating new pose...')
                        break
                if valid == False:
                    continue
                # If everything is ok, add the object to the list
                final_pose = p
                final_pose.orientation.x = 0
                final_pose.orientation.y = 0
                p1_xyz = np.array([p.position.x, p.position.y, p.position.z])
                p2_xyz = np.array([final_pose.position.x, final_pose.position.y, final_pose.position.z-5])

                orientation = p2_xyz - p1_xyz
                norm_orientation = np.linalg.norm(orientation)
                orientation = orientation / norm_orientation

                ray_origins = np.array([p1_xyz])
                ray_directions = np.array([orientation])
                # Check the collision with the mesh
                collisions, _, _ = self.mesh_collision.ray.intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions)
                # If there is a collision, move the object to the closest collision, floor.
                closest_collision_to_p1 = self.getClosestCollision(collisions, p1_xyz)
                final_pose.position.z = final_pose.position.z - closest_collision_to_p1
                final_poses.append(final_pose)
                object_names.append(object['name'])
                break # Goes to the next object
        
        print(f'{Fore.GREEN}Generated poses for {len(final_poses)} objects{Style.RESET_ALL}')
        manager.append(final_poses)
        manager.append(object_names)

        return manager

    def generatePath(self, model_name , final_pose=None, init_pose=None):
        if init_pose == None:
            initial_pose = self.getPose(model_name).pose
        else:
            initial_pose = init_pose

        if final_pose == None:
            final_pose = self.generateRandomPose()

            while True:
                xyz_initial = np.array(
                    [initial_pose.position.x, initial_pose.position.y, initial_pose.position.z])
                xyz_final = np.array(
                    [final_pose.position.x, final_pose.position.y, final_pose.position.z])
                l2_dst = np.linalg.norm(xyz_final - xyz_initial)

                # if final pose is close to the initial or there is collision, choose another final pose
                if l2_dst < 1.5 or self.checkCollision(initial_pose=initial_pose, final_pose=final_pose):
                    final_pose = self.generateRandomPose()
                else:
                    break

        # compute n_steps based on l2_dist
        n_steps = int(l2_dst / self.dbf)

        print('using n_steps of: ', n_steps)

        step_poses = []  # list of tuples
        rx, ry, rz = tf.transformations.euler_from_quaternion(
            [initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w])
        pose_initial_dct = {'x': initial_pose.position.x,
                            'y': initial_pose.position.y,
                            'z': initial_pose.position.z,
                            'rx': rx,
                            'ry': ry,
                            'rz': rz}

        rx, ry, rz = tf.transformations.euler_from_quaternion(
            [final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w])
        pose_final_dct = {'x': final_pose.position.x,
                          'y': final_pose.position.y,
                          'z': final_pose.position.z,
                          'rx': rx,
                          'ry': ry,
                          'rz': rz}

        x_step_var = (pose_final_dct['x'] - pose_initial_dct['x']) / n_steps
        y_step_var = (pose_final_dct['y'] - pose_initial_dct['y']) / n_steps
        z_step_var = (pose_final_dct['z'] - pose_initial_dct['z']) / n_steps
        rx_step_var = (pose_final_dct['rx'] - pose_initial_dct['rx']) / n_steps
        ry_step_var = (pose_final_dct['ry'] - pose_initial_dct['ry']) / n_steps
        rz_step_var = (pose_final_dct['rz'] - pose_initial_dct['rz']) / n_steps

        for i in range(n_steps):
            dct = {'x': pose_initial_dct['x'] + (i + 1) * x_step_var,
                   'y': pose_initial_dct['y'] + (i + 1) * y_step_var,
                   'z': pose_initial_dct['z'] + (i + 1) * z_step_var,
                   'rx': pose_initial_dct['rx'] + (i + 1) * rx_step_var,
                   'ry': pose_initial_dct['ry'] + (i + 1) * ry_step_var,
                   'rz': pose_initial_dct['rz'] + (i + 1) * rz_step_var}
            pose = data2pose(dct)
            step_poses.append(pose)

        return step_poses

    def getPose(self , model_name):
        return self.get_model_state_service(model_name, 'world')

    def setPose(self, model_name , pose = None):
        # If no pose is given, set the pose to the resting pose
        if pose == None:
            pose = self.resting_pose
        req = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        req.model_state.model_name = model_name
        req.model_state.pose.position.x = pose.position.x
        req.model_state.pose.position.y = pose.position.y
        req.model_state.pose.position.z = pose.position.z
        req.model_state.pose.orientation.x = pose.orientation.x
        req.model_state.pose.orientation.y = pose.orientation.y
        req.model_state.pose.orientation.z = pose.orientation.z
        req.model_state.pose.orientation.w = pose.orientation.w
        req.model_state.reference_frame = 'world'
        try:
            response = self.set_state_service(req.model_state)
            if response.status_message == "SetModelState: model does not exist":
                self.spawnModel(model_name, pose)
  
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
     
    def spawnModel(self,object_name,object_position):
        spawn_model = SpawnModelRequest()
        spawn_model.model_name = object_name
        spawn_model.model_xml = open(f'{self.path}/models_3d/localbot/Objects/{object_name}/model.sdf', 'r').read()
        spawn_model.robot_namespace = ''
        spawn_model.initial_pose = object_position
        self.spawn_model_service(spawn_model)

    def deleteModel(self,object_name):
        delete_model = DeleteModelRequest()
        delete_model.model_name = object_name
        self.delete_model_service(delete_model)

    def generateLights(self, n_steps, random):
        lights = []
        if random:
            lights = [np.random.uniform(
                low=self.att_min, high=self.att_max) for _ in range(n_steps)]
        else:
            initial_light = self.att_initial
            final_light = np.random.uniform(
                low=self.att_min, high=self.att_max)

            step_light = (final_light - initial_light) / n_steps

            for i in range(n_steps):
                lights.append(initial_light + (i + 1) * step_light)

            self.att_initial = final_light
        return lights

    def setLight(self, attenuation_quadratic):

        for light in self.lights:
            pose = Pose()
            pose.position = Point(light['pose'][0],light['pose'][1],light['pose'][2])
            pose.orientation = Quaternion(0,0,0,1)

            # Creates the light message to be sent to the gazebo service
            light_msg = SetLightPropertiesRequest()
            light_msg.light_name = light['name']
            light_msg.cast_shadows = True
            light_msg.diffuse = ColorRGBA(0.8,0.8,0.8,1)
            light_msg.specular = ColorRGBA(0.2,0.2,0.2,1)
            light_msg.attenuation_constant = 0.1
            light_msg.attenuation_linear = 0.01
            light_msg.attenuation_quadratic = attenuation_quadratic
            light_msg.pose = pose
            light_msg.direction = Vector3(1e-6,1e-6,-1)
            self.modify_light(light_msg) 


    def getSunAzimuth(self , n_steps , random): 
        # Definition of a time range of simulation
        time_change = datetime.timedelta(minutes=20*n_steps)
        self.final_time = self.initial_time + time_change

        # Definition of a time range of simulation
        times = pd.date_range(self.initial_time, self.final_time, inclusive='left', freq=f'20T', tz=self.site.tz)

        solpos_nrel = pvlib.solarposition.get_solarposition(times, self.site.latitude, self.site.longitude, self.site.altitude, method='nrel_numpy')

        self.initial_time = self.final_time

        return solpos_nrel['azimuth'], solpos_nrel['zenith'] , times

    
    def setSunLight(self, roll = 0, pitch = 0, yaw = 0 , time=0):
        if time.hour > 18  or time.hour < 6:
            roll = 0
            pitch = 0
            yaw = 0

        # Required to make the service call to gazebo
        pose = Pose()
        pose.position = Point(0,0,5)
        orientation = tf.transformations.quaternion_from_euler(roll*math.pi/180, pitch*math.pi/180, yaw*math.pi/180)
        pose.orientation = Quaternion(0,0,0,1)
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        # Creates the light message to be sent to the gazebo service
        light = SetLightPropertiesRequest()
        light.light_name = 'sun'
        light.cast_shadows = True
        light.diffuse = ColorRGBA(0.8,0.8,0.8,1)
        light.specular = ColorRGBA(0.2,0.2,0.2,1)
        light.attenuation_constant = 0.9
        light.attenuation_linear = 0.01
        light.attenuation_quadratic = 0.001
        light.pose = pose
        light.direction = Vector3(1e-6,1e-6,-1)

        # Creates the service and sends the light message
        self.modify_light(light)


    def checkCollision(self, initial_pose, final_pose):
        if self.use_collision is False:
            print('not using COLLISIONS.')
            return False

        initial_pose.position.x

        p1_xyz = np.array(
            [initial_pose.position.x, initial_pose.position.y, initial_pose.position.z])
        p1_quat = np.array([initial_pose.orientation.x, initial_pose.orientation.y,
                           initial_pose.orientation.z, initial_pose.orientation.w])

        p2_xyz = np.array(
            [final_pose.position.x, final_pose.position.y, final_pose.position.z])
        p2_quat = np.array([final_pose.orientation.x, final_pose.orientation.y,
                           final_pose.orientation.z, final_pose.orientation.w])

        dist_p1_to_p2 = np.linalg.norm(p2_xyz-p1_xyz)

        print(
            f' {Fore.BLUE} Checking collision... {Fore.RESET} between {p1_xyz} and {p2_xyz}')

        orientation = p2_xyz - p1_xyz
        norm_orientation = np.linalg.norm(orientation)
        orientation = orientation / norm_orientation

        ray_origins = np.array([p1_xyz])
        ray_directions = np.array([orientation])

        collisions, _, _ = self.mesh_collision.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)

        closest_collision_to_p1 = self.getClosestCollision(collisions, p1_xyz)

        # compare the closest collision with the position of p2
        if closest_collision_to_p1 < dist_p1_to_p2:
            # collision
            print(f'{Fore.RED} Collision Detected. {Fore.RESET}')
            return True
        else:
            # no collision

            # check if p2 camera viewpoint if close to a obstacle.
            orientation = R.from_quat(p2_quat).as_matrix()[:, 0]
            norm_orientation = np.linalg.norm(orientation)
            orientation = orientation / norm_orientation

            ray_origins = np.array([p2_xyz])
            ray_directions = np.array([orientation])

            collisions, _, _ = self.mesh_collision.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions)

            closest_collision_to_p2 = self.getClosestCollision(
                collisions, p2_xyz)

            if closest_collision_to_p2 < self.min_cam_dist:

                print(
                    f'{Fore.YELLOW} Final Pose is too close to a obstacle. {Fore.RESET}')
                return True
            else:
                print(f'{Fore.GREEN} NO Collision Detected {Fore.RESET}')
                return False
            
    def checkInsideMesh(self,points):
        result = self.mesh_collision.contains(points)
        return result.all()

    def checkCollisionVis(self, initial_pose, final_pose):
        if self.use_collision is False:
            print('not using COLLISIONS.')
            return False
        # load mesh
        # TODO #83 this should not be hardcoded
        mesh = trimesh.load(
            '/home/danc/models_3d/santuario_collision/Virtudes_Chapel.dae', force='mesh')

        initial_pose.position.x

        p1_xyz = np.array(
            [initial_pose.position.x, initial_pose.position.y, initial_pose.position.z])
        p1_quat = np.array([initial_pose.orientation.x, initial_pose.orientation.y,
                           initial_pose.orientation.z, initial_pose.orientation.w])

        p2_xyz = np.array(
            [final_pose.position.x, final_pose.position.y, final_pose.position.z])
        p2_quat = np.array([final_pose.orientation.x, final_pose.orientation.y,
                           final_pose.orientation.z, final_pose.orientation.w])

        dist_p1_to_p2 = np.linalg.norm(p2_xyz-p1_xyz)

        print(
            f' {Fore.BLUE} Checking collision... {Fore.RESET} between {p1_xyz} and {p2_xyz}')

        orientation = p2_xyz - p1_xyz
        norm_orientation = np.linalg.norm(orientation)
        orientation = orientation / norm_orientation

        ray_origins = np.array([p1_xyz])
        ray_directions = np.array([orientation])

        collisions, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)

        closest_collision_to_p1 = self.getClosestCollision(collisions, p1_xyz)

        # compare the closest collision with the position of p2
        if closest_collision_to_p1 < dist_p1_to_p2:
            # collision
            print(f'{Fore.RED} Collision Detected. {Fore.RESET}')
            return 1
        else:
            # no collision

            # check if p2 camera viewpoint if close to a obstacle.
            orientation = R.from_quat(p2_quat).as_matrix()[:, 0]
            norm_orientation = np.linalg.norm(orientation)
            orientation = orientation / norm_orientation

            ray_origins = np.array([p2_xyz])
            ray_directions = np.array([orientation])

            collisions, _, _ = mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions)

            closest_collision_to_p2 = self.getClosestCollision(
                collisions, p2_xyz)

            if closest_collision_to_p2 < self.min_cam_dist:

                print(
                    f'{Fore.YELLOW} Final Pose is too close to a obstacle. {Fore.RESET}')
                return 0.5
            else:
                print(f'{Fore.GREEN} NO Collision Detected {Fore.RESET}')
                return 0

    def generatePathViz(self, final_pose):

        initial_pose = self.getPose().pose

        xyz_initial = np.array(
            [initial_pose.position.x, initial_pose.position.y, initial_pose.position.z])
        xyz_final = np.array(
            [final_pose.position.x, final_pose.position.y, final_pose.position.z])
        l2_dst = np.linalg.norm(xyz_final - xyz_initial)

        # compute n_steps based on l2_dist
        n_steps = int(l2_dst / self.dbf)

        print('using n_steps of: ', n_steps)

        step_poses = []  # list of tuples
        rx, ry, rz = tf.transformations.euler_from_quaternion(
            [initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w])
        pose_initial_dct = {'x': initial_pose.position.x,
                            'y': initial_pose.position.y,
                            'z': initial_pose.position.z,
                            'rx': rx,
                            'ry': ry,
                            'rz': rz}

        rx, ry, rz = tf.transformations.euler_from_quaternion(
            [final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w])
        pose_final_dct = {'x': final_pose.position.x,
                          'y': final_pose.position.y,
                          'z': final_pose.position.z,
                          'rx': rx,
                          'ry': ry,
                          'rz': rz}

        x_step_var = (pose_final_dct['x'] - pose_initial_dct['x']) / n_steps
        y_step_var = (pose_final_dct['y'] - pose_initial_dct['y']) / n_steps
        z_step_var = (pose_final_dct['z'] - pose_initial_dct['z']) / n_steps
        rx_step_var = (pose_final_dct['rx'] - pose_initial_dct['rx']) / n_steps
        ry_step_var = (pose_final_dct['ry'] - pose_initial_dct['ry']) / n_steps
        rz_step_var = (pose_final_dct['rz'] - pose_initial_dct['rz']) / n_steps

        for i in range(n_steps):
            dct = {'x': pose_initial_dct['x'] + (i + 1) * x_step_var,
                   'y': pose_initial_dct['y'] + (i + 1) * y_step_var,
                   'z': pose_initial_dct['z'] + (i + 1) * z_step_var,
                   'rx': pose_initial_dct['rx'] + (i + 1) * rx_step_var,
                   'ry': pose_initial_dct['ry'] + (i + 1) * ry_step_var,
                   'rz': pose_initial_dct['rz'] + (i + 1) * rz_step_var}
            pose = data2pose(dct)
            step_poses.append(pose)

        return step_poses

    def getClosestCollision(self, collisions, p1_xyz):
        closest_collision_to_p1 = np.inf

        for position_collision in collisions:
            dist_collision_p1 = np.linalg.norm(position_collision - p1_xyz)
            if dist_collision_p1 < closest_collision_to_p1:
                closest_collision_to_p1 = dist_collision_p1
        return closest_collision_to_p1

    def saveFrame(self):
        self.save_dataset.saveFrame()

    def getFrameIdx(self):
        return self.save_dataset.frame_idx
