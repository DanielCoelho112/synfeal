import subprocess
import numpy as np
import transforms3d as t3d
from scipy.spatial.transform import Rotation  
import tf


def execute(cmd, blocking=True, verbose=True):
    """ @brief Executes the command in the shell in a blocking or non-blocking manner
        @param cmd a string with teh command to execute
        @return
    """
    if verbose:
        print("Executing command: " + cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if blocking:  # if blocking is True:
        for line in p.stdout.readlines():
            if verbose:
                print
                line,
            p.wait()

# Example 4x4 transformation matrix (replace this with your actual matrix)
# transform_matrix = np.array([[0.581802,-0.522479,-0.623315,20.7373],
#                              [0.813015,0.394935,0.427823,10.6052],
#                              [0.0226401,-0.755673,0.654557,1.44539],
#                              [0, 0, 0, 1]])

transform_matrix = np.array([[0.73205,-0.17063,0.65954,20.74645],
                            [0.64091,-0.15570,-0.75166,10.61321],
                            [0.23095,0.97296,-0.00462,1.44828],
                            [0,0,0,1]])

# Extract translation (x, y, z)
translation = transform_matrix[:3, 3]
x, y, z = translation
print(translation)

# Extract rotation as a matrix
rotation_matrix = transform_matrix[:3, :3]
print(rotation_matrix)
# Convert rotation matrix to RPY (Roll, Pitch, Yaw)
# rpy = t3d.euler.mat2euler(rotation_matrix, axes='szyx')
# roll, pitch, yaw = rpy

angles = tf.transformations.euler_from_matrix(rotation_matrix, axes = 'szxy')

print(angles)
# r =  Rotation.from_matrix(rotation_matrix)
# angles = r.as_euler("zyx",degrees=False)
# print(angles)

# print(f'roslaunch synfeal_bringup bringup_camera.launch x_pos:=20.7373 y_pos:=10.6052 z_pos:=1.44539 R_pos:={angles[0]} P_pos:={angles[1]} Y_pos:={angles[2]}')

execute(cmd=f'roslaunch synfeal_bringup bringup_camera.launch x_pos:={translation[0]} y_pos:={translation[1]} z_pos:={translation[2]} R_pos:={angles[0]} P_pos:={angles[1]} Y_pos:={angles[2]}')

# Print the results
# print("Translation (x, y, z):", x, y, z)
# # print("Rotation (RPY):", np.degrees(roll), np.degrees(pitch), np.degrees(yaw))

# print("Rotation (RPY):", roll, pitch, yaw)

