"""
    usage: python3 preview_o3d.py path/to/file/something.pcd
"""
import open3d as o3d
import sys

pcd = o3d.io.read_point_cloud(sys.argv[1])
o3d.visualization.draw_geometries([pcd])