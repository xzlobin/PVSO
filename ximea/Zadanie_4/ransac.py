import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from preview_o3d import filter_nan

pcd = o3d.io.read_point_cloud("./ximea/Zadanie_4/Pcs/kitchen/Rf17.pcd")
pcd, _ = filter_nan(pcd)

#Normals Computation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)

#3D Shape Detection with RANSAC
max_planes = 10
indexes = np.array(range(max_planes))
colors = plt.get_cmap("tab20")(indexes / max_planes)
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

segment_models={}
segments={}
rest=pcd
for i in tqdm(range(max_planes)):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)

o3d.visualization.draw_geometries([segments[i] for i in range(max_planes)])