#!/usr/bin/env python3

# --------------------------------------------------
# Adapted from http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28Python%29
# -------------------------------------------------

from functools import partial
import rospy
import tf2_ros
import geometry_msgs.msg
from gazebo_msgs.msg import ModelStates

def callbackModelStatesReceived(msg, tf_broadcaster):
    childs = msg.name
    pose = msg.pose
    world = 'world'
    now = rospy.Time.now()

    # the gazebo has several models, so we have to pick the one we want
    if 'localbot' in childs: 
        
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
    rospy.init_node('model_states_to_tf')
    rospy.Subscriber("/gazebo/model_states_throttle", ModelStates,
                     partial(callbackModelStatesReceived, tf_broadcaster=tf2_ros.TransformBroadcaster()))
    rospy.spin()

if __name__ == '__main__':
    main()
