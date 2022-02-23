#!/usr/bin/env python3

# --------------------------------------------------
# Miguel Riem Oliveira.
# August 2021.
# Adapted from http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28Python%29
# -------------------------------------------------
import math
from functools import partial
import rospy
import tf_conversions  # Because of transformations
import tf2_ros
import geometry_msgs.msg
from gazebo_msgs.msg import ModelState, ModelStates

def callbackModelStatesReceived(msg, tf_broadcaster):
    # print('received data ' + str(msg))
    print('Received message')

    now = rospy.Time.now()
    world_link = rospy.remap_name('world')  # assume
    for idx, name in enumerate(msg.name):
        link = name + '/odom'  # TODO: find a way to get root link of an urdf

        # print(name)
        pose = msg.pose[idx]

        names = ['localbot']
        if name in names:  # do this only for my player
            print(pose)

            # Create the transformation from /world to /odom (Identity)
            # transform = geometry_msgs.msg.TransformStamped()
            # transform.header.frame_id = world_link
            # transform.child_frame_id = 'gazebo_base_link'
            # transform.header.stamp = now
            # transform.transform.rotation.w = 1
            # tf_broadcaster.sendTransform(transform)
            #
            # Create the transformation /odom to /base_footprint
            transform = geometry_msgs.msg.TransformStamped()
            transform.header.frame_id = world_link
            transform.child_frame_id = '/base_footprint_gazebo'  # TODO should not be hardcoded
            transform.header.stamp = now
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z

            transform.transform.rotation.x = pose.orientation.x
            transform.transform.rotation.y = pose.orientation.y
            transform.transform.rotation.z = pose.orientation.z
            transform.transform.rotation.w = pose.orientation.w
            tf_broadcaster.sendTransform(transform)


def main():
    rospy.init_node('model_states_to_tf')  # initialize the ros node
    rospy.Subscriber("/gazebo/model_states", ModelStates,
                     partial(callbackModelStatesReceived, tf_broadcaster=tf2_ros.TransformBroadcaster()))

    rospy.spin()


if __name__ == '__main__':
    main()
