import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

# reuse function to filter nan values 
from preview_o3d import filter_nan

# determine the right path to save results
script_folder = os.path.dirname(os.path.realpath(__file__))

file = os.path.join(script_folder, "results", "Rf17_ransac.pcd")

# load point cloud and filter out NaN values if needed
pcd = o3d.io.read_point_cloud(file)
pcd, _ = filter_nan(pcd)

# Epsilon for DBScan
DBScan_eps = 0.03

# Clustering using DBSCan
labels = np.array(pcd.cluster_dbscan(eps=DBScan_eps, min_points=5))
mlabel = labels.max()
print(f"DBScan: {mlabel + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (mlabel if mlabel > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

os.makedirs(os.path.join(script_folder, "results"), exist_ok=True)
name = os.path.basename(file).replace(".pcd", "_dbscan_clusters.pcd")
o3d.io.write_point_cloud(os.path.join(script_folder, "results", name), pcd)

o3d.visualization.draw_geometries([pcd])


# Amount of clusters for K-Means
KMeans_n = 10

# Clustering using K-Means
kmeans = KMeans(n_clusters=KMeans_n, random_state=0, n_init="auto")
print(f"K-Means: set {KMeans_n} clusters")

# Taking points from pcd
pcd_npdata = np.array(pcd.points)
kmeans = kmeans.fit(pcd_npdata)

colors_km = plt.get_cmap("tab20")(kmeans.labels_ / KMeans_n)

# coloring points
pcd.colors = o3d.utility.Vector3dVector(colors_km[:, :3])

name = os.path.basename(file).replace(".pcd", "_kmeans_clusters.pcd")
o3d.io.write_point_cloud(os.path.join(script_folder, "results", name), pcd)

o3d.visualization.draw_geometries([pcd])