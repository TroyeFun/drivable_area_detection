import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import draw_geometries


def detect_drivable_area_from_pcd(pcd, voxel_size):
    # 1. Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2. Estimate the normals of the downsampled point cloud
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 3. Apply the RANSAC plane segmentation
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=voxel_size, ransac_n=3, num_iterations=1000)

    # 4. Extract the inliers
    inlier_cloud = pcd_down.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    # 5. Extract the outliers
    outlier_cloud = pcd_down.select_by_index(inliers, invert=True)

    # 6. Apply the DBSCAN clustering
    labels = np.array(outlier_cloud.cluster_dbscan(eps=voxel_size * 2, min_points=10, print_progress=True))

    # 7. Extract the clusters
    max_label = labels.max()
    print("point cloud has {} clusters".format(max_label + 1))
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 8. Visualize the results
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def detect_drivable_area_from_mesh(mesh, distance_threshold):
    # 1. Downsample the mesh
    mesh.compute_vertex_normals()
    mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * 0.1))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])

    # 2. Apply the RANSAC plane segmentation
    pcd = mesh.sample_points_poisson_disk(10000)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)

    # 3. find vertex that is close to the plane
    vertex = np.asarray(mesh.vertices)
    dist_vertex2plane = np.abs(np.dot(vertex, plane_model[:3]) + plane_model[3])
    inlier_vertex = np.where(dist_vertex2plane < distance_threshold)[0]

    # 4. paint ground vertices
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertex_colors[inlier_vertex] = [1, 0, 0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # 5. Visualize the results
    o3d.visualization.draw_geometries([mesh])


def ray_ground_filter(pcd):
    pass


if __name__ == "__main__":
    distance_threshold = 0.3
    mesh = o3d.io.read_triangle_mesh("test_data/3760.ply")
    pcd = mesh.sample_points_poisson_disk(10000)
    # detect_drivable_area_from_pcd(pcd, distance_threshold)
    detect_drivable_area_from_mesh(mesh, distance_threshold)