# pcd_aug.py

import os
import numpy as np
import argparse
import sys, os, io, traceback, json, subprocess, glob, re
import open3d as o3d
import laspy

# add noise to spatial coordinates (simulate sensor jitters/noise)
def noising(data, prob):
    if np.random.uniform() < prob:
        data[:, :3] += np.random.normal(0, 0.01, size=data[:, :3].shape)
        
# rotates coords by random angles around the z-axis
def rotating(data, prob):
    if np.random.uniform() < prob:
        angle = np.random.uniform(0, 360)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                    [np.sin(angle), np.cos(angle)]])
        data[:,:2] = data[:,:2].dot(rotation_matrix)

# changes the scale of the coordinates by random factor btwn 0.9 and 1.1
def scaling(data, prob):
    if np.random.uniform() < prob:
        data[:, :3] *= np.random.uniform(0.9, 1.1)

# translates/shifts x and y coords by small amount of gaussian noise
def translating(data, prob):
    if np.random.uniform() < prob:
        data[:, :2] += np.random.normal(0, 0.01, size=data[:, :2].shape)

# add noise to RGB values (vary the colour slightly by ~-5 to 5)
def rgb_noising(data, prob):
    if np.random.uniform() < prob:
        data[:, 3:] += np.random.randint(-5, 5, size=data[:, 3:].shape)
        data[:, 3:] = np.clip(data[:, 3:], 0, 255)

# add noise to RGB channels to simulate changing light conditions 
def rgb_light_effect(data, prob):
    if np.random.uniform() < prob:
        data[:, 3:] += np.random.normal(0, 20, size=data[:, 3:].shape)
        data[:, 3:] = np.clip(data[:, 3:], 0, 255)

def main():
    parser = argparse.ArgumentParser(description='Perform data augmentation on a given dataset.')
    parser.add_argument('--input_folder', type=str, default="./input", help='the path to the input folder')
    parser.add_argument('--output_folder', type=str, default="./output", help='the path to the output folder')
    parser.add_argument('--noising', type=float, default=0.5, help='the probability of performing noising (default: 0.50)')
    parser.add_argument('--scaling', type=float, default=0.5, help='the probability of performing scaling (default: 0.50)')
    parser.add_argument('--translating', type=float, default=0.0, help='the probability of performing translating (default: 0.50)')
    parser.add_argument('--rotating', type=float, default=0.5, help='the probability of performing rotating (default: 0.50)')
    parser.add_argument('--rgb_noising', type=float, default=0.5, help='the probability of performing RGB noising (default: 0.50)')
    parser.add_argument('--rgb_light_effect', type=float, default=0.5, help='the probability of performing RGB light effect (default: 0.50)')
    parser.add_argument('--aug_num', type=int, default=50, help='number of the augmentation of each class')
    args = parser.parse_args()
	
    os.makedirs(args.output_folder, exist_ok=True)

    # fetch files from the input folder
    input_files = os.listdir(args.input_folder)
    for fname in input_files:
        path = os.path.join(args.input_folder, fname)

        name, ext = fname.rsplit('.', 1)
       
        # check pcd file (assumes binary/ascii)
        pcd_o3d = o3d.io.read_point_cloud(path)
        pcd_points = np.asarray(pcd_o3d.points)
        pcd_colors = np.asarray(pcd_o3d.colors)

        # Ensure colors are scaled [0,255] and stacked with XYZ
        if pcd_colors.shape[0] == 0:
            # No color data — fill with white (255,255,255)
            pcd_colors = np.full_like(pcd_points, 255)

        pcd_colors = (pcd_colors * 255).astype(np.int32) if pcd_colors.max() <= 1.0 else pcd_colors.astype(np.int32)
        pcd_data = np.hstack((pcd_points, pcd_colors))

        pcd_count = len(pcd_data)
        if pcd_count == 0:
            continue

        # perform augmentations 
        noising(pcd_data, args.noising)
        scaling(pcd_data, args.scaling)
        translating(pcd_data, args.translating)
        rotating(pcd_data, args.rotating)
        rgb_noising(pcd_data, args.rgb_noising)
        rgb_light_effect(pcd_data, args.rgb_light_effect)

        # convert to PCD format 
        pcd_data[:,3:] = pcd_data[:,3:].astype(int)

        output_file_path = os.path.join(args.output_folder, name + ".pcd")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(pcd_data[:,3:] / 255.0)
        o3d.io.write_point_cloud(output_file_path, pcd)

        print("Saved {} points to {}".format(pcd_count, output_file_path))

        las_header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header=las_header)

        las.x = pcd_data[:, 0]
        las.y = pcd_data[:, 1]
        las.z = pcd_data[:, 2]

        if pcd_data.shape[1] > 3:
            las.red = pcd_data[:, 3].astype(np.uint16)
            las.green = pcd_data[:, 4].astype(np.uint16)
            las.blue = pcd_data[:, 5].astype(np.uint16)

        las.write(os.path.join(args.output_folder, name + ".las"))

        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
