#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R

#rotate_y90 = R.from_euler('y', 90, degrees=True).as_matrix()
rotate_y90 = R.from_euler('x', 40, degrees=True).as_quat()

matrix = R.from_quat(rotate_y90).as_matrix()
print(matrix)
print(matrix[:,0])