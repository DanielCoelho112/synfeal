#!/usr/bin/env python3

import os
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
    path = os.environ.get("SYNFEAL_DATASET")
    spawn_model = SpawnModelRequest()
    spawn_model.model_name = object_name
    spawn_model.model_xml = open(f'{path}/models_3d/localbot/Objects/{object_name}/model.sdf', 'r').read()
    spawn_model.robot_namespace = ''
    spawn_model.initial_pose = object_position
    spawn_model_service(spawn_model)
    rospy.sleep(5)



def main():
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-model3d', '--model3d_config', type=str,
                        required=True, help='object poses config to use')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))    

    # check if model3d file exists.
    # get absolute path of localbot_core.
    synfeal_collection_path = rospkg.RosPack().get_path('synfeal_collection')
    model3d_config_path = f'{synfeal_collection_path}/model3d_config/{args["model3d_config"]}.yaml'

    if exists(model3d_config_path):
        with open(model3d_config_path) as f:
            model3d_config = yaml.load(f, Loader=SafeLoader)
            # important to associate with the dataset to be created.
            model3d_config['name'] = args["model3d_config"]
    else:
        print(f'{Fore.RED} model3d config path ({model3d_config_path}) does not exist! \n Aborting... {Fore.RESET}')
        exit(0)

    objects = model3d_config['objects']

    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # init ros node.
    rospy.init_node(f"object_spawn_visualizer")
    rate = rospy.Rate(10)

    row = 0
    col = 0
    ############################################################
    ######################     path      #######################
    ############################################################
    while not rospy.is_shutdown():
        print(Fore.GREEN + f'Spawning objects, total number of objects are {len(objects)}' + Style.RESET_ALL)
        for idx,object in enumerate(objects):
            print(f'Placing object {object["name"]}')
            p = Pose()
            p.position.x = col
            p.position.y = row
            p.position.z = 0
            col += 1
            object_name = f'{object["name"]}'
            spawnModel(object_name,p,spawn_model_service)
            if col == 9:
                row += 1
                col = 0


        rate.sleep()
        print(Fore.GREEN + "Done" + Style.RESET_ALL)
        break


if __name__ == "__main__":
    main()
