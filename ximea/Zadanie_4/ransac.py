import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from preview_o3d import filter_nan

file = "./ximea/Zadanie_4/Pcs/kitchen/Rf17.pcd"

script_folder = os.path.dirname(os.path.realpath(__file__))
pcd = o3d.io.read_point_cloud(file)
pcd, _ = filter_nan(pcd)

#Normals Computation
#pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)

#3D Shape Detection with RANSAC
max_planes = 20
indexes = np.array(range(max_planes))
colors = plt.get_cmap("tab20")(indexes / max_planes)
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

segments=[]
rest=pcd
for i in tqdm(range(max_planes)):
    colors = plt.get_cmap("tab20")(i)
    _, inliers = rest.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
    segments.append(rest.select_by_index(inliers))
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)

rest.paint_uniform_color([0,0,0])

os.makedirs(os.path.join(script_folder, "results"), exist_ok=True)
name = os.path.basename(file).replace(".pcd", "_ransac.pcd")

filtered = segments[0]
for s in segments:
    filtered += s

o3d.io.write_point_cloud(os.path.join(script_folder, "results", name), filtered)
o3d.visualization.draw_geometries([segments[i] for i in range(max_planes)]+[rest])