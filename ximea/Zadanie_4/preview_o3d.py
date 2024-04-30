"""
    usage: python3 preview_o3d.py path/to/file/something.pcd
"""
import open3d as o3d
import numpy as np
import sys

def filter_nan(pcd):
    """
    Function to filter out NaN values from the point cloud.
    Returns filtered (pcd, amount of NaNs points).
    """
    pcd_points = np.array(pcd.points)
    pcd_colors = np.array(pcd.colors)
    row_selector = ~np.isnan(pcd_points).any(axis=1)
    if row_selector.all():
        return pcd, 0
    pcd_points = pcd_points[row_selector, :]
    pcd_colors = pcd_colors[row_selector, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd, (~row_selector).sum()

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(sys.argv[1])
    pcd, nan_count = filter_nan(pcd)
    print(pcd)
    print(np.array(pcd.points))
    print(f"Deleted {nan_count} NaN points")
    o3d.visualization.draw([pcd])
