import os
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.geometry import TriangleMesh as TM
from open3d.visualization import draw_geometries


def preprocess_mesh(mesh):
    mesh = TM(mesh)
    # mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * 0.1))
    mesh = mesh.filter_smooth_laplacian(20)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    # mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
    return mesh


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


def get_ground_plane_from_mesh(mesh, ransac_dist_thrd, drive_dist_thrd):

    # 2. Apply the RANSAC plane segmentation
    pcd = mesh.sample_points_poisson_disk(10000)
    plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_dist_thrd, ransac_n=3, num_iterations=1000)
    return plane_model  # [a, b, c, d]: ax + by + cz + d = 0


def paint_plane(mesh, plane_model, dist_thrd):
    mesh = o3d.geometry.TriangleMesh(mesh)
    vertex = np.asarray(mesh.vertices)
    dist_vertex2plane = np.abs(np.dot(vertex, plane_model[:3]) + plane_model[3])
    inlier_vertex = np.where(dist_vertex2plane < dist_thrd)[0]
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertex_colors[inlier_vertex] = [1, 0, 0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def ray_ground_filter(pcd):
    pass


def get_vertex_triangle_neighbor_list(mesh):
    vert2trg = [[] for _ in range(len(mesh.vertices))]
    for tid, trg in enumerate(mesh.triangles):
        for vid in trg:
            vert2trg[vid].append(tid)
    return vert2trg


def get_triangle_centers(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    triangle_centers = np.mean(verts[tris], axis=1)
    return triangle_centers


def get_triangle_adjacency(mesh):
    vert2trg = get_vertex_triangle_neighbor_list(mesh)
    adjacency_list = [None for _ in range(len(mesh.triangles))]
    for tid in range(len(mesh.triangles)):
        trg = mesh.triangles[tid]
        neighbors = []
        for vid in trg:
            neighbors += vert2trg[vid]
        neighbors = list(set(neighbors))
        neighbors.remove(tid)
        adjacency_list[tid] = neighbors
    return adjacency_list


def paint_triangle(mesh, tid, color=(1, 0, 0)):
    for vid in mesh.triangles[tid]:
        mesh.vertex_colors[vid] = color


def abs_slope_check(mesh, tid, max_slope_rad):
    normal = mesh.triangle_normals[tid]
    ez = np.array([0, 0, 1])
    slope_rad = np.arccos(np.dot(normal, ez))
    return slope_rad <= max_slope_rad


def rel_slope_check(mesh, tid1, tid2, max_slope_rad):
    normal1 = mesh.triangle_normals[tid1]
    normal2 = mesh.triangle_normals[tid2]
    slope_rad = np.arccos(np.dot(normal1, normal2))
    return slope_rad <= max_slope_rad


def degree2rad(degree):
    return np.pi * degree / 180


def detect_ground(mesh, start_point, max_abs_slope_angle, max_rel_slope_angle, paint=True):
    mesh = TM(mesh)
    trg_adj_list = get_triangle_adjacency(mesh)
    tri_centers = get_triangle_centers(mesh)
    max_abs_slope_rad = degree2rad(max_abs_slope_angle)
    max_rel_slope_rad = degree2rad(max_rel_slope_angle)

    drivable_trgs = [False] * len(mesh.triangles)

    start_point = np.array(start_point).reshape(1, 3)
    dist_to_start = np.linalg.norm(tri_centers - start_point, axis=1)
    start_tid = np.argmin(dist_to_start)
    queue = [start_tid]
    drivable_trgs[start_tid] = True
    bfs_seq = []
    while queue:
        tid = queue[0]
        queue = queue[1:]
        if paint:
            paint_triangle(mesh, tid)
        bfs_seq.append(tid)
        for neigh_tid in trg_adj_list[tid]:
            if drivable_trgs[neigh_tid]:
                continue
            if not abs_slope_check(mesh, neigh_tid, max_abs_slope_rad):
                continue
            if not rel_slope_check(mesh, tid, neigh_tid, max_rel_slope_rad):
                continue
            queue.append(neigh_tid)
            drivable_trgs[neigh_tid] = True
    return mesh, drivable_trgs, bfs_seq


def paint_mesh_incrementally(mesh, bfs_seq):
    """Copy the following text to clipboard, and paste it in the console of Open3D Visualizer
    {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 51.0, 25.399999999999999, 4.2000000000000002 ],
			"boundingbox_min" : [ -0.059999999999999998, -18.800774000000001, -2.0 ],
			"field_of_view" : 60.0,
			"front" : [ -0.90763089025171184, -0.057936705080019447, 0.415751735132105 ],
			"lookat" : [ 24.626496372118194, 0.68516761492840161, -5.7363715400712492 ],
			"up" : [ 0.40881059807817505, 0.10283041080047317, 0.90680747764537706 ],
			"zoom" : 0.33999999999999964
		}
	],
	"version_major" : 1,
	"version_minor" : 0
    }
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    mesh = TM(mesh)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    print('paste the text')
    time.sleep(3)
    vis.poll_events()
    vis.update_renderer()

    img_cnt = 0
    save_dir = 'test_data/drivable_painting'
    os.makedirs(save_dir, exist_ok=True)
    try:
        for i, tid in enumerate(bfs_seq):
            paint_triangle(mesh, tid)
            if i % 10000 == 0:
                vis.update_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image("{}/{}.png".format(save_dir, img_cnt))
                img_cnt += 1
    except KeyboardInterrupt:
        pass
    vis.destroy_window()
    return mesh



if __name__ == "__main__":
    ransac_dist_thrd = 0.3
    drive_dist_thrd = 0.3
    max_abs_slope_angle = 25  # in degree
    max_rel_slope_angle = 40  # in degree

    # raw_mesh = o3d.io.read_triangle_mesh("test_data/000038.ply")
    # start_point = np.array([8, 0, -1.5])  # 000038.ply
    raw_mesh = o3d.io.read_triangle_mesh("test_data/00203.ply")
    start_point = np.array([0, 8, -1.5])  # 00203.ply

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    mesh = preprocess_mesh(raw_mesh)

    # pcd = mesh.sample_points_poisson_disk(10000)
    # detect_drivable_area_from_pcd(pcd, distance_threshold)

    # plane_model = get_ground_plane_from_mesh(mesh, ransac_dist_thrd, drive_dist_thrd)
    # plane_model = [0.00255629, -0.02898346,  0.99957662,  1.57113454]
    # mesh_painted = paint_plane(mesh, plane_model, drive_dist_thrd)
    # o3d.visualization.draw_geometries([mesh_painted, origin])
    mesh, drivable_trgs, bfs_seq = detect_ground(mesh, start_point, max_abs_slope_angle, max_rel_slope_angle, paint=True)
    mesh = paint_mesh_incrementally(raw_mesh, bfs_seq)
    draw_geometries([mesh, origin])
    import ipdb; ipdb.set_trace()