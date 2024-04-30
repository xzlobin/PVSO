import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# reuse function to filter nan values 
from preview_o3d import filter_nan

# since script can be run in different working directory we have to
# determine the right path to save results
script_folder = os.path.dirname(os.path.realpath(__file__))

# path to pcd to work with
#file = os.path.join(script_folder, "PCs", "kitchen", "Rf17.pcd")
file = os.path.join(script_folder, "PCs", "output.pcd")

# load point cloud and filter out NaN values if needed
pcd = o3d.io.read_point_cloud(file)
pcd, _ = filter_nan(pcd)

# estimating normals, used just for visualisation of pcd
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)

# max amout of planes to be found. 
# affects the quality of filtration
max_planes = 20

# distance threshold for ransac to consider point to be a part of a plane
distance_thr = 0.03

# planar segments of the pcd
segments=[]
# segments with original colors
segments_orig=[]

# reverse accamulator of the pcd points
rest=pcd

# cycle to filter out the noise points.
# after interations 'rest will contain the outliers of the planes
for i in tqdm(range(max_planes)):
    colors = plt.get_cmap("tab20")(i)
    # ransac pass on the rest of pcd
    _, inliers = rest.segment_plane(distance_threshold=distance_thr,ransac_n=3,num_iterations=1000)
    # saving the segment found
    segments.append(rest.select_by_index(inliers))
    segments_orig.append(o3d.geometry.PointCloud(segments[i]))
    # coloring segments to different colors
    segments[i].paint_uniform_color(list(colors[:3]))
    # selecting only the points that have not been choosen by ransac
    rest = rest.select_by_index(inliers, invert=True)

# the noise will be colored in black for the visualisation
rest.paint_uniform_color([0,0,0])

# creating dir to save the filtered pcd and the noise pcd
os.makedirs(os.path.join(script_folder, "results"), exist_ok=True)
name = os.path.basename(file).replace(".pcd", "_ransac_clusters.pcd")
rest_name = os.path.basename(file).replace(".pcd", "_ransac_noise.pcd")
name_orig = os.path.basename(file).replace(".pcd", "_ransac.pcd")

# concatenate the segments in one point cloud
filtered = segments[0]
for s in segments:
    filtered += s
filtered_orig = segments_orig[0]
for s in segments_orig:
    filtered_orig += s

# re-color noisy point cloud, just to save it in red
rest_red = o3d.geometry.PointCloud(rest)
rest_red.paint_uniform_color([1,0,0])

# writing results down
o3d.io.write_point_cloud(os.path.join(script_folder, "results", name), filtered)
o3d.io.write_point_cloud(os.path.join(script_folder, "results", rest_name), rest_red)
o3d.io.write_point_cloud(os.path.join(script_folder, "results", name_orig), filtered_orig)

# visualising the in an interactive window
o3d.visualization.draw_geometries(segments + [rest])