import argparse
import math
import os
import random
import sys
import h5py
import multiprocessing
import subprocess
import time

import numpy as np
import open3d as o3d
from tqdm import tqdm


PARTNETSIM_COLOR_MAP = {
    0: (255.0, 0, 0),
    1: (0, 255.0, 0),
    2: (0, 0, 255.0),
    3: (255.0, 0, 255.0),
}

def get_random_color():
    r = float(random.randint(0, 255)) / 255
    g = float(random.randint(0, 255)) / 255
    b = float(random.randint(0, 255)) / 255
    return (r, g, b)

def gt_mesh_subprocess(scene_id):
    command = f"cd ../scripts/visualization; python visualize_gt_mesh.py -s {scene_id}"
    subprocess.Popen(command, shell=True)

def visualize_window(pcd, w_id):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='gt' if w_id == 0 else 'prediction', width=960, height=540, left=0 if w_id == 0 else 960, top=0)
    vis.add_geometry(pcd)

    vis.run()

    vis.destroy_window()


def visualize_pc(model_id, prediction_path, points, colors, normals, gtsemantic_ids, mesh_flag, partnetsim_path, unlabelled_flag):
    with open(f"{prediction_path}/{model_id}.txt") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    pcd_colors = colors
    instanceFileNames = []
    labelIndexes = []
    confidenceScores = []
    predicted_mask_list = []
    for i in lines:
        splitedLine = i.split()
        instanceFileNames.append(os.path.join(prediction_path, splitedLine[0]))
        labelIndexes.append(splitedLine[1])
        confidenceScores.append(splitedLine[2])

    for instanceFileName in instanceFileNames:
        predicted_mask_list.append(np.loadtxt(instanceFileName, dtype=bool))

    if not unlabelled_flag:
        for index, predicted_mask in enumerate(predicted_mask_list):
            semanticIndex = labelIndexes[index]
            for pointIndex, color in enumerate(pcd_colors):
                    if predicted_mask[pointIndex] == True:
                        pcd_colors[pointIndex] = PARTNETSIM_COLOR_MAP[int(semanticIndex)]
    else:
        for index, predicted_mask in enumerate(predicted_mask_list):
            semanticIndex = labelIndexes[index]
            instance_color = get_random_color()
            print(instance_color)
            for pointIndex, color in enumerate(pcd_colors):
                    if predicted_mask[pointIndex] == True:
                        pcd_colors[pointIndex] = instance_color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    #pcd.normals = o3d.utility.Vector3dVector(normals)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #o3d.visualization.draw_geometries([pcd, coordinate])

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points)

    for index, color in enumerate(colors):
        colors[index] = PARTNETSIM_COLOR_MAP[int(gtsemantic_ids[index])]


    pcd_gt.colors = o3d.utility.Vector3dVector(colors)

    p = multiprocessing.Process(name='gt_window', target=visualize_window, args=(pcd_gt, 0, ))
    p2 = multiprocessing.Process(name='pred_window', target=visualize_window, args=(pcd, 1, ))
    #if mesh_flag:
    #    p3 = multiprocessing.Process(name='gt_mesh_visualization', target=gt_mesh_subprocess, args=(model_id,))

    p.start()
    p2.start()
    #if mesh_flag:
    #    p3.start()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./output/PartNetSim/PointGroup/testing-old/inference/val/predictions/instance', 
                        help='specify path to predictions')
    
    parser.add_argument('-s', '--scene_id', type=str, default=None, 
                        help='specify model id from val set')
    
    parser.add_argument('-d', '--dataset', type=str, default='../data/dataset_color_normal_triangles_corrected', 
                        help='specify path to dataset')
    
    parser.add_argument('-m', '--mesh', type=bool, default=True, 
                        help='flag for visualizing gt mesh')
    
    parser.add_argument('-p', '--partnetsim_path', type=str, default='../data/dataset', 
                        help='specify path to dataset')
    
    parser.add_argument('-u', '--unlabelled', type=bool, default=False, 
                        help='specify whether to treat predictions as unlabelled(instance) or semantic')
    
    
    args = parser.parse_args()

    downsample_data = h5py.File(f"{args.dataset}/downsample.h5", "a")

    points = downsample_data["points"]
    instance_ids = downsample_data["instance_ids"]
    colors = downsample_data["colors"]
    normals = downsample_data["normals"]
    downsample_model_ids = downsample_data["model_ids"]
    semantic_ids = downsample_data["semantic_ids"]


    num_models = downsample_model_ids.shape[0]
    model_idx_map = {}
    for i in range(num_models):
        model_idx_map[downsample_model_ids[i].decode("utf-8")] = i

    
    cur_points = points[model_idx_map[args.scene_id]]
    cur_colors = colors[model_idx_map[args.scene_id]]
    #cur_colors = [(255, 0, 255) * cur_points.shape[0]]
    cur_normals = normals[model_idx_map[args.scene_id]]
    cur_instance_ids = instance_ids[model_idx_map[args.scene_id]]
    cur_semantic_ids = semantic_ids[model_idx_map[args.scene_id]]

    visualize_pc(args.scene_id, args.input_path, cur_points, cur_colors, cur_normals, cur_semantic_ids, args.mesh, args.partnetsim_path, args.unlabelled)