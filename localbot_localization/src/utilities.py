import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from localbot_core.src.utilities import *
from scipy.spatial.transform import Rotation as R

def normalize_quat(x, p=2, dim=1):
    """
    Divides a tensor along a certain dim by the Lp norm
    :param x: 
    :param p: Lp norm
    :param dim: Dimension to normalize along
    :return: 
    """
    
    if torch.is_tensor(x):
        # x.shape = (N,4)
        xn = x.norm(p=p, dim=dim) # computes the norm: 1xN
        x = x / xn.unsqueeze(dim=dim)
    
    else: # numpy
        xn = np.linalg.norm(x)
        x = x/xn
        
    return x

def process_pose(pose):
    quat_unit = normalize_quat(pose[:,3:])
    return torch.cat((pose[:,:3], quat_unit), dim=1)


def compute_position_error(pred, targ):
    pred = pred[:3]
    targ = targ[:3]
    
    return mean_squared_error(pred, targ, squared=False) # RMSE

def compute_rotation_error(pred, targ): 
    
    ## first way: using quaternions:
    ## https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames and https://stackoverflow.com/questions/23260939/distance-or-angular-magnitude-between-two-quaternions and https://stackoverflow.com/questions/20798056/magnitude-of-rotation-between-two-quaternions
    #pred = pred[3:]
    #targ = targ[3:]
    # pred = Quaternion(pred)
    # targ = Quaternion(targ)
    # deltaq = q_cross(pred, q_inv(targ))
    # return 2 * atan2(np.linalg.norm(deltaq[:3]), deltaq[3]) # angle in rads
    
    ## second way: using rodrigues (like ATOM) --> better because angle ranges from 0 to pi (whereas with quaterions ranges from 0 to 2pi)
    ## https://github.com/lardemua/atom/blob/284b7943e467e53a3258de6f673cf852b07654cb/atom_evaluation/scripts/camera_to_camera_evalutation.py#L290
    pred_matrix = poseToMatrix(pred)
    targ_matrix = poseToMatrix(targ)
    
    delta = np.dot(np.linalg.inv(pred_matrix), targ_matrix)
    deltaR = matrixToRodrigues(delta[0:3, 0:3])
    
    return np.linalg.norm(deltaR)
    
    
def projectToCamera(intrinsic_matrix, distortion, width, height, pts):
    """
    Projects a list of points to the camera defined transform, intrinsics and distortion
    :param intrinsic_matrix: 3x3 intrinsic camera matrix
    :param distortion: should be as follows: (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])
    :param width: the image width
    :param height: the image height
    :param pts: a list of point coordinates (in the camera frame) with the following format: np array 4xn or 3xn
    :return: a list of pixel coordinates with the same length as pts
    """

    _, n_pts = pts.shape

    # Project the 3D points in the camera's frame to image pixels
    # From https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    pixs = np.zeros((2, n_pts), dtype=np.float)

    k1, k2, p1, p2, k3 = distortion
    # fx, _, cx, _, fy, cy, _, _, _ = intrinsic_matrix
    # print('intrinsic=\n' + str(intrinsic_matrix))
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    x = pts[0, :]
    y = pts[1, :]
    z = pts[2, :]

    dists = np.linalg.norm(pts[0:3, :], axis=0)  # compute distances from point to camera
    xl = np.divide(x, z)  # compute homogeneous coordinates
    yl = np.divide(y, z)  # compute homogeneous coordinates
    r2 = xl ** 2 + yl ** 2  # r square (used multiple times bellow)
    xll = xl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * xl * yl + p2 * (r2 + 2 * xl ** 2)
    yll = yl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (r2 + 2 * yl ** 2) + 2 * p2 * xl * yl
    pixs[0, :] = fx * xll + cx
    pixs[1, :] = fy * yll + cy

    # Compute mask of valid projections
    valid_z = z > 0
    valid_xpix = np.logical_and(pixs[0, :] >= 0, pixs[0, :] < width)
    valid_ypix = np.logical_and(pixs[1, :] >= 0, pixs[1, :] < height)
    valid_pixs = np.logical_and(valid_z, np.logical_and(valid_xpix, valid_ypix))
    return pixs, valid_pixs, dists


def synthesize_pose(pose1, pose2):
    """
    synthesize pose between pose1 and pose2
    pose1: 4x4
    pose2: 4x4
    """
    pos1 = pose1[:3,3]
    rot1 = pose1[:3,:3]
    
    pos2 = pose2[:3,3]
    rot2 = pose2[:3,:3]
    
    # rot3x3 to euler angles
    rot1_euler = R.from_matrix(rot1).as_euler('xyz', degrees=False)
    rot2_euler = R.from_matrix(rot2).as_euler('xyz', degrees=False)
    
    pos3 = (pos1 + pos2) / 2
    rot3_euler = (rot1_euler + rot2_euler) / 2
    
    rot3 = R.from_euler('xyz', rot3_euler, degrees=False).as_matrix()
    
    pose3 = np.empty(shape=(4,4))
    pose3[:3,:3] = rot3
    pose3[:3,3] = pos3
    
    return pose3

