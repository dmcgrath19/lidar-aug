# pcd_feat_extr.py

import sys, os, io, traceback, json, subprocess, argparse, glob, re, numpy as np, sklearn, matplotlib.pyplot as plt, open3d as o3d
import laspy, copy, pandas as pd
import trimesh
from jakteristics import FEATURE_NAMES, extension, las_utils, utils
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
import ifcopenshell.util.representation
from ifcopenshell import geom

data_dir = Path(__file__).parent / "data"
  

def visualise_feature(xyz, features, feature_name, save_path=None):
    idx = FEATURE_NAMES.index(feature_name)
    vals = features[:, idx]
    norm = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
    colors = plt.cm.jet(norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd])


def compute_features(input_path, output_path, feature_name= 'planarity'):
    '''
        FEATURE_NAMES = [
            "eigenvalue_sum",
            "omnivariance",
            "eigenentropy",
            "anisotropy",
            "planarity",
            "linearity",
            "PCA1",
            "PCA2",
            "surface_variation",
            "sphericity",
            "verticality",
            "nx",
            "ny",
            "nz",
            "number_of_neighbors",
            "eigenvalue1",
            "eigenvalue2",
            "eigenvalue3",
            "eigenvector1x",
            "eigenvector1y",
            "eigenvector1z",
            "eigenvector2x",
            "eigenvector2y",
            "eigenvector2z",
            "eigenvector3x",
            "eigenvector3y",
            "eigenvector3z",
        ]   
    '''

    # read point coordinates from file, compute, write features
    xyz = las_utils.read_las_xyz(input_path)
    features = extension.compute_features(xyz, 0.1, feature_names=FEATURE_NAMES)
    las_utils.write_with_extra_dims(input_path, output_path, features, FEATURE_NAMES)

    # visualise feature
    visualise_feature(xyz, features, feature_name=feature_name, save_path="feature_visualization.jpg")

def main():
    global data_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="sample.las", type=str, required=False)
    parser.add_argument('--output', default="output.las", type=str, required=False)
    args = parser.parse_args()
    
    compute_features(args.input, args.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()