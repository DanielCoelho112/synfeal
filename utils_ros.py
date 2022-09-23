import copy
import math
import tf
import rospy
import os
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import *
from std_msgs.msg import Header, ColorRGBA
from synfeal_collection.src.pypcd import PointCloud


def write_pcd(filename, msg, mode='binary'):
    
    pc = PointCloud.from_msg(msg)
    pc.save_pcd(filename, compression=mode)
    
def read_pcd(filename):

    if not os.path.isfile(filename):
        raise Exception("[read_pcd] File does not exist.")
    pc = PointCloud.from_path(filename)

    return pc
    

def data2pose(data):
    
    if type(data) is str:
        data = list(data)
        lst_data = [i for i in data if i!=','] # remove ','
        data = {'x'  : lst_data[0], 
                'y'  : lst_data[1], 
                'z'  : lst_data[2],
                'rx' : lst_data[3],
                'ry' : lst_data[4], 
                'rz' : lst_data[5]}
        
    quaternion = tf.transformations.quaternion_from_euler(data['rx'], data['ry'], data['rz'])
    #quaternion = R.from_euler('xyz',[[data['rx'], data['ry'], data['rz']]], degrees=False).as_quat()
    
    p = Pose()
    p.position.x = data['x']
    p.position.y = data['y']
    p.position.z = data['z']
    
    p.orientation.x = quaternion[0]
    p.orientation.y = quaternion[1]
    p.orientation.z = quaternion[2]
    p.orientation.w = quaternion[3]
        
    return p


def createArrowMarker(pose, color):
        
    pose_marker = copy.deepcopy(pose)
    matrix_quaternion_marker = pose_marker[3:]
    #matrix_quaternion_marker = R.from_quat(pose_marker[3:]).as_matrix()
    # rotate_y90 = R.from_euler('y', -90, degrees=True).as_matrix()
    # matrix_quaternion_marker = np.dot(
    #     matrix_quaternion_marker, rotate_y90)
    # quaternion_marker = R.from_matrix(
    #     matrix_quaternion_marker).as_quat()

    marker = Marker(header=Header(
        frame_id="world", stamp=rospy.Time.now()))
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.3
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = color[-1]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose.orientation.x = matrix_quaternion_marker[0]
    marker.pose.orientation.y = matrix_quaternion_marker[1]
    marker.pose.orientation.z = matrix_quaternion_marker[2]
    marker.pose.orientation.w = matrix_quaternion_marker[3]
    marker.pose.position.x = pose[0]
    marker.pose.position.y = pose[1]
    marker.pose.position.z = pose[2]
    marker.ns = 'final_pose'
    marker.id = 1
    
    return marker

def getFrustumMarkerArray(w, h, f_x, f_y, Z_near, Z_far, frame_id, ns, color, alpha=0.9, thickness=0.005, lifetime=False):
    # big help from https: // github.com/ros-visualization/rviz/issues/925
    marker_array = MarkerArray()

    # ------------------------------------
    # Define view frustum points
    # ------------------------------------
    fov_x = 2 * math.atan2(w, (2 * f_x))
    fov_y = 2 * math.atan2(h, (2 * f_y))

    x_n = math.tan(fov_x / 2) * Z_near
    y_n = math.tan(fov_y / 2) * Z_near

    x_f = math.tan(fov_x / 2) * Z_far
    y_f = math.tan(fov_y / 2) * Z_far

    points = [Point(-x_n, y_n, Z_near),
              Point(x_n, y_n, Z_near),
              Point(x_n, -y_n, Z_near),
              Point(-x_n, -y_n, Z_near),
              Point(-x_f, y_f, Z_far),
              Point(x_f, y_f, Z_far),
              Point(x_f, -y_f, Z_far),
              Point(-x_f, -y_f, Z_far)]
        # ------------------------------------
    # Define wireframe
    # ------------------------------------

    color_rviz = ColorRGBA(r=color[0]/2, g=color[1]/2, b=color[2]/2, a=1.0)
    marker = Marker(ns=ns+'_wireframe', type=Marker.LINE_LIST, action=Marker.ADD, header=Header(frame_id=frame_id),
                    color=color_rviz)
    if lifetime:
        marker.lifetime=rospy.Duration(0)

    marker.scale.x = thickness  # line width
    marker.pose.orientation.w = 1.0

    # marker line points
    marker.points.append(points[0])
    marker.points.append(points[1])

    marker.points.append(points[1])
    marker.points.append(points[2])

    marker.points.append(points[2])
    marker.points.append(points[3])

    marker.points.append(points[3])
    marker.points.append(points[0])

    marker.points.append(points[0])
    marker.points.append(points[4])

    marker.points.append(points[1])
    marker.points.append(points[5])

    marker.points.append(points[2])
    marker.points.append(points[6])

    marker.points.append(points[3])
    marker.points.append(points[7])

    marker.points.append(points[4])
    marker.points.append(points[5])

    marker.points.append(points[5])
    marker.points.append(points[6])

    marker.points.append(points[6])
    marker.points.append(points[7])

    marker.points.append(points[7])
    marker.points.append(points[4])

    marker_array.markers.append(copy.deepcopy(marker))

    # ------------------------------------
    # Define filled
    # ------------------------------------
    color_rviz = ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
    marker = Marker(ns=ns+'_filled', type=Marker.TRIANGLE_LIST, action=Marker.ADD, header=Header(frame_id=frame_id),
                    color=color_rviz)
    if lifetime:
        marker.lifetime=rospy.Duration(0)

    marker.scale.x = 1  # line width
    marker.scale.y = 1  # line width
    marker.scale.z = 1  # line width
    marker.pose.orientation.w = 1.0

    # marker triangles of the lateral face of the frustum pyramid
    marker.points.append(points[1])
    marker.points.append(points[2])
    marker.points.append(points[6])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[1])
    marker.points.append(points[6])
    marker.points.append(points[5])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[0])
    marker.points.append(points[4])
    marker.points.append(points[3])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[3])
    marker.points.append(points[4])
    marker.points.append(points[7])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[0])
    marker.points.append(points[1])
    marker.points.append(points[5])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[0])
    marker.points.append(points[4])
    marker.points.append(points[5])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[3])
    marker.points.append(points[2])
    marker.points.append(points[6])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker.points.append(points[3])
    marker.points.append(points[6])
    marker.points.append(points[7])
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)
    marker.colors.append(color_rviz)

    marker_array.markers.append(copy.deepcopy(marker))

    return marker_array
