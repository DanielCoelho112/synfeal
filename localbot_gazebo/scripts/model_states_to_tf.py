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
    childs = msg.name
    pose = msg.pose
    world = rospy.remap_name('world') 
    
    if 'localbot' in childs:
        now = rospy.Time.now()
        idx = childs.index('localbot')
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = world
        transform.child_frame_id = '/base_footprint'
        transform.header.stamp = now
        transform.transform.translation.x = pose[idx].position.x
        transform.transform.translation.y = pose[idx].position.y
        transform.transform.translation.z = pose[idx].position.z

        transform.transform.rotation.x = pose[idx].orientation.x
        transform.transform.rotation.y = pose[idx].orientation.y
        transform.transform.rotation.z = pose[idx].orientation.z
        transform.transform.rotation.w = pose[idx].orientation.w
        tf_broadcaster.sendTransform(transform)


def main():
    rospy.init_node('model_states_to_tf')  # initialize the ros node
    rospy.Subscriber("/gazebo/model_states", ModelStates,
                     partial(callbackModelStatesReceived, tf_broadcaster=tf2_ros.TransformBroadcaster()))

    rospy.spin()


if __name__ == '__main__':
    main()
