#!/usr/bin/env python3

# stdlib
import sys
import argparse

# 3rd-party
import trimesh
import numpy as np
import time

def main():
    # parser = argparse.ArgumentParser(description='Data Collector')
    # parser.add_argument('-m', '--mode', type=str, default='interactive',
    #                     help='interactive/automatic_random_path/automatic_path')
    
    #mesh = trimesh.creation.icosphere()
    #mesh = trimesh.exchange.dae.load_collada('/home/danc/models_3d/santuario_collision/Virtudes_Chapel.dae')
    start_time = time.time()
    mesh = trimesh.load('/home/danc/models_3d/santuario_collision/Virtudes_Chapel.dae', force='mesh')
    

    
    
    p1 = np.array([0,0,0])
    p2 = np.array([4,0,0])
    dp1p2 = np.linalg.norm(p2-p1)
    ori = p2 - p1
    norm_ori =norm = np.linalg.norm(ori)
    ori = ori / norm_ori
    
    
    # create some rays
    #ray_origins = np.array([[0, 0, 0]])                       
    #ray_directions = np.array([[0, 1, 0]])
    
    ray_origins = np.array([p1])
    ray_directions = np.array([ori])
    
    
    # check out the docstring for intersects_location queries
    print(mesh.ray.intersects_location.__doc__)
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)

    print('The rays hit the mesh at coordinates:\n', locations)

    print(time.time() - start_time)

    # get the first intersection
    dists_to_p1 = []
    
    for dist in locations:
        dists_to_p1.append(np.linalg.norm(dist - p1))
        
        
    print(dists_to_p1)
    
    closest_collision = min(dists_to_p1)
    
    print(closest_collision)
    print(dp1p2)
    
    if closest_collision < dp1p2:
        print('COLISSION')
    else:
        print('SAFE')
    
    # compare the first intersection with p2 (in terms of distances)
    
    
if __name__ == "__main__":
    main()
