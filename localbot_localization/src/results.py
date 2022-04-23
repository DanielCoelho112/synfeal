from localbot_core.src.utilities import matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import pandas as pd
import os


class LocalBotResults():
    def __init__(self, results_path):
        self.path = f'{os.environ["HOME"]}/results/localbot/{results_path}'
        self.nframes = int(sum(f.endswith('.txt') for f in os.listdir(self.path))/2)
        self.csv = pd.read_csv(f'{self.path}/errors.csv')
        
    def __getitem__(self, index):
        
        # load pose
        matrix_predicted = np.loadtxt(f'{self.path}/frame-{index:05d}.predicted.pose.txt', delimiter=',')
        matrix_real = np.loadtxt(f'{self.path}/frame-{index:05d}.real.pose.txt', delimiter=',')
        
        quaternion_real = matrixToQuaternion(matrix_real)
        quaternion_real = normalize_quat(quaternion_real)
        xyz_real = matrixToXYZ(matrix_real)
        pose_real = np.append(xyz_real, quaternion_real)        
        
        quaternion_predicted = matrixToQuaternion(matrix_predicted)
        quaternion_predicted = normalize_quat(quaternion_predicted)
        xyz_predicted = matrixToXYZ(matrix_predicted)
        pose_predicted = np.append(xyz_predicted, quaternion_predicted)
                
        return pose_real, pose_predicted
                

    def __len__(self):
        return self.nframes
    
    def getErrorsArrays(self):
        pos_error_array = self.csv.iloc[:-1]['position_error (m)'].to_numpy()
        rot_error_array = self.csv.iloc[:-1]['rotation_error (rads)'].to_numpy()
        return pos_error_array, rot_error_array

    def updateCSV(self):
        self.csv.to_csv(f'{self.path}/errors.csv', index=False, float_format='%.5f')
        
        


#results = LocalBotResults('test1')
#print(len(results))


