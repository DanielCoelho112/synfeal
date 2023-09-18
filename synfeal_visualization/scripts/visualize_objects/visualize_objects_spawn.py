#!/usr/bin/env python3

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel , SpawnModelRequest 
import sys
import argparse
import rospkg
import rospy
import yaml
from os.path import exists
from yaml.loader import SafeLoader
from colorama import Fore , Style

def spawnModel(object_name,object_position,spawn_model_service):
    spawn_model = SpawnModelRequest()
    spawn_model.model_name = object_name
    synfeal_visualization_path = rospkg.RosPack().get_path('synfeal_visualization')
    spawn_model.model_xml = open(f'{synfeal_visualization_path}/model3d_config/sphere.sdf', 'r').read()
    spawn_model.robot_namespace = ''
    spawn_model.initial_pose = object_position
    spawn_model_service(spawn_model)



def main():
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-pc', '--poses_config', type=str,
                        required=True, help='object poses config to use')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))    

    # check if model3d file exists.
    # get absolute path of localbot_core.
    synfeal_collection_path = rospkg.RosPack().get_path('synfeal_collection')
    poses_config_path = f'{synfeal_collection_path}/model3d_config/{args["poses_config"]}_poses.yaml'
    if exists(poses_config_path):
        with open(poses_config_path) as f:
            poses_config = yaml.load(f, Loader=SafeLoader)
        object_poses = poses_config['object_poses']
    else:
        print(f"{Fore.RED}ERROR: {poses_config_path} not found.{Style.RESET_ALL}")
        sys.exit(1)

    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # init ros node.
    rospy.init_node(f"object_spawn_visualizer")
    rate = rospy.Rate(10)

    ############################################################
    ######################     path      #######################
    ############################################################
    while not rospy.is_shutdown():
        print(Fore.GREEN + f'Spawning objects, total object spawn points are {len(object_poses)}' + Style.RESET_ALL)
        for idx,pose in enumerate(object_poses):
            p = Pose()
            p.position.x = pose['pose'][0]
            p.position.y = pose['pose'][1]
            p.position.z = pose['pose'][2]
            object_name = f'object_{idx}'
            spawnModel(object_name,p,spawn_model_service)


        rate.sleep()
        print(Fore.GREEN + "Done" + Style.RESET_ALL)
        break


if __name__ == "__main__":
    main()
