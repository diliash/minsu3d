import argparse
import math
import os
import random
import sys
import h5py
import random
import trimesh
import tempfile
import pyglet as gl
import open3d as o3d

sys.path.append('../../..')

from pathlib import Path

from opmotion import (
    CatBox,
    PartnetsimParser,
)
from Helper3D import SampleSurfaceFromTrimeshScene, getURDF

import numpy as np
from tqdm import tqdm
from minsu3d.minsu3d.util.pc import write_ply_rgb_face

import io
from PIL import Image


PARTNETSIM_COLOR_MAP = {
    0: (0, 0, 0),
    1: (174, 199, 232),
    2: (152, 223, 138),
    3: (31, 119, 180),
}

PARTNETSIM_COLOR_MAP_REVERSE = {
    (0.0, 0.0, 0.0): 0,
    (174, 199, 232): 1, 
    (152, 223, 138): 2,
    (31, 119, 180): 3,
}

def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]


def get_random_rgb_colors(num):
    rgb_colors = [get_random_color() for _ in range(num)]
    return rgb_colors


def generate_colored_ply(args, predicted_mask_list, labelIndexes, points, colors, triangles, cur_geometry_map, 
                         rgb_inst_ply):
    print(args.mode)
    if args.mode == "semantic":
        for index, predicted_mask in enumerate(predicted_mask_list):
            semanticIndex = labelIndexes[index]
            # confidence = confidenceScores[index]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = PARTNETSIM_COLOR_MAP[int(semanticIndex)]
    elif args.mode == "instance":
        color_list = get_random_rgb_colors(len(labelIndexes))
        random.shuffle(color_list)
        for index, predicted_mask in enumerate(predicted_mask_list):
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = color_list[index]
    #write_ply_rgb_face(points, colors, indices, rgb_inst_ply)

    
    urdf, controller = getURDF(os.path.join(args.scans, args.scene_id, "mobility.urdf"))
    mesh = urdf.getMesh()

    print(np.unique(triangles).shape)
    colors = colors.astype(int)

    face_colors = np.empty((1, 3))
    
    
    for key, geometry in mesh.geometry.items():
        print(geometry)
        print(geometry.visual)
        print(geometry.visual.defined)
        geometry.visual = geometry.visual.to_color()
        print(geometry.visual)
        print(geometry.visual.kind)
        print(geometry.visual.defined)
        
        exit()
        geometry_triangles = triangles[cur_geometry_map == key]
        for triangle in np.unique(geometry_triangles):
            relevant_indexes = np.where(geometry_triangles == triangle)[0]
            relevant_colors = colors[relevant_indexes] 
            relevant_labels = [PARTNETSIM_COLOR_MAP_REVERSE[tuple(color)] for color in relevant_colors]
            face_colors = np.append(face_colors, np.expand_dims(np.asarray(PARTNETSIM_COLOR_MAP[np.bincount(relevant_labels).argmax()]), axis=0), axis=0)
    
    face_colors = np.delete(face_colors, 0, 0)
    
    all_geometry = np.array([], dtype=int)
    for key, geometry in mesh.geometry.items():
        print(geometry.faces)
        all_geometry = np.append(all_geometry, geometry.faces)
    print(np.unique(all_geometry))
    print(np.shape(face_colors))
    print(np.shape(mesh.triangles))
    print(mesh.triangles_node)
    print(np.shape(mesh.triangles_node))
    
    
    
    return 0



def generate_single_ply(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # alignment_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}.txt')
    pred_sem_file = os.path.join(args.predict_dir, f'{args.scene_id}.txt')

    # define where to output the ply file
    rgb_inst_ply = os.path.join(args.output_dir, f'{args.scene_id}.ply')

    downsample_data = h5py.File("../../../data/dataset_color_normal_triangles/downsample.h5", "a")

    points = downsample_data["points"]
    instance_ids = downsample_data["instance_ids"]
    colors = downsample_data["colors"]
    normals = downsample_data["normals"]
    downsample_model_ids = downsample_data["model_ids"]
    semantic_ids = downsample_data["semantic_ids"]
    face_indexes = downsample_data["face_indexes"]
    barycentric_coordinates = downsample_data["barycentric_coordinates"]
    geometry_map = downsample_data["geometry_map"]

    num_models = downsample_model_ids.shape[0]
    model_idx_map = {}
    for i in range(num_models):
        model_idx_map[downsample_model_ids[i].decode("utf-8")] = i
    
    cur_points = np.asarray(points[model_idx_map[args.scene_id]])
    cur_colors = np.asarray(colors[model_idx_map[args.scene_id]])
    cur_normals = np.asarray(normals[model_idx_map[args.scene_id]])
    cur_instance_ids = np.asarray(instance_ids[model_idx_map[args.scene_id]])
    cur_triangles = np.asarray(face_indexes[model_idx_map[args.scene_id]])
    cur_geometry_map = geometry_map[model_idx_map[args.scene_id]]
    cur_geometry_map = np.asarray([item.decode("utf-8") for item in cur_geometry_map])
    

    with open(pred_sem_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    instanceFileNames = []
    labelIndexes = []
    confidenceScores = []
    predicted_mask_list = []
    for i in lines:
        splitedLine = i.split()
        instanceFileNames.append(os.path.join(args.predict_dir, splitedLine[0]))
        labelIndexes.append(splitedLine[1])
        confidenceScores.append(splitedLine[2])

    for instanceFileName in instanceFileNames:
        predicted_mask_list.append(np.loadtxt(instanceFileName, dtype=bool))

    generate_colored_ply(args, predicted_mask_list, labelIndexes, cur_points, cur_colors, cur_triangles, cur_geometry_map, 
                             rgb_inst_ply)


def generate_pred_inst_ply(args):
    metadata_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/partnetsim/metadata')
    scene_ids_file = os.path.join(metadata_path, f'partnetsim_{args.split}.txt')
    args.scans = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/partnetsim/dataset')

    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    for scene_id in tqdm(scene_ids):
        args.scene_id = scene_id
        generate_single_ply(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict_dir', type=str,
                        default='../../output/PartNetSim/PointGroup/poingroup_partnetsim_elastic_off/inference/val/predictions/instance',
                        help='the directory of the predictions. Eg:"../../output/ScanNet/SoftGroup/test/predictions/instance"')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['test', 'val'],
                        help='specify the split of data: val | test')
    parser.add_argument('-m', '--mode', type=str, default='semantic', choices=['semantic', 'instance'],
                        help='specify instance or semantic mode: semantic | instance')
    parser.add_argument('-o', '--output_dir', type=str, default='./output_ply',
                        help='the directory of the output ply')
    parser.add_argument('-i', '--id', type=str, default=None, help='specify one scene id for ply generation')
    
    
    args = parser.parse_args()
    args.rgb_file_dir = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/partnetsim', args.split)

    args.output_dir = os.path.join(args.output_dir, "color")
    args.output_dir = os.path.join(args.output_dir, args.mode)

    if args.id:
        args.scene_id = args.id
        args.scans = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/partnetsim/dataset')
        generate_single_ply(args)
    else:
        generate_pred_inst_ply(args)
