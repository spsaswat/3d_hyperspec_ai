import numpy as np
import sys
import cv2
import open3d as o3d
import os
import glob
from natsort import natsorted
import json


def undistort_depth(depth, K, dist):
    """
    Undistortion for depth image
    cv2.undistort uses bilinar interpolation
    but it causes artifacts for boundary of depth image.
    Nearest Neighbor is preferred.
    This function provides NN like undistortion
    :param depth: input depth image
    :param K: camera intrinsic calibration matrix
    :param dist: distortion paratmeters
    :return: a distorted image of the same shape as depth
    """

    # Intrinsic calibration matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    undistorted = np.zeros_like(depth)
    h, w = depth.shape
    # u,v creates a mesh that together forms the coordinate sets of the whole image
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h), (w, 1)).T
    u, v = _undistort_pixel_opencv(u, v, fx, fy, cx, cy, dist)
    # round distorted coordinate to integer coordinates
    v, u = np.rint(v).astype(np.int), np.rint(u).astype(np.int)

    # Make valid mask
    v_valid = np.logical_and(0 <= v, v < h)
    u_valid = np.logical_and(0 <= u, u < w)
    uv_valid = np.logical_and(u_valid, v_valid)
    uv_invalid = np.logical_not(uv_valid)

    # Fiil stub
    v[v < 0] = 0
    v[(h - 1) < v] = h - 1
    u[u < 0] = 0
    u[(w - 1) < u] = w - 1

    # Copy depth value
    # Similar to Nearest Neighbor
    undistorted[v, u] = depth

    # 0 for invalid
    undistorted[uv_invalid] = 0

    return undistorted


# https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
def _undistort_pixel_opencv(u, v, fx, fy, cx, cy, dist):
    # https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
    # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385
    if np.shape(dist)[0] == 8:
        k1, k2, p1, p2, k3, k4, k5, k6 = dist
    elif np.shape(dist)[0] == 5:
        k1, k2, p1, p2, k3 = dist
        k4, k5, k6 = 0, 0, 0
    x0 = (u - cx) / fx
    y0 = (v - cy) / fy
    x = x0
    y = y0
    # Compensate distortion iteratively
    # 5 is from OpenCV code.
    # I don't know theoritical rationale why 5 is enough...
    max_iter = 5
    for j in range(max_iter):
        r2 = x * x + y * y
        icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / \
                 (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - deltaX) * icdist
        y = (y0 - deltaY) * icdist

    u_ = x * fx + cx
    v_ = y * fy + cy

    return u_, v_


def clean_pcd(pcd: o3d.geometry.PointCloud, nb_neighbors=10, std_ratio=0.6, nb_points=10, radius=10):
    'Remove outliers from the point cloud based on distance curve and number of neighbours in ball'
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.4)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=5)
    return pcd


def color_icp(source, target):
    """
    registration using color icp algorithm
    :param source: source point cloud
    :param target: target point cloud
    :return: Merged point clouds in phase with the target point cloud, transformation matrix of the source point cloud,
            fitness between two models and rmse.
    """
    voxel_radius = [30, 10, 5]
    max_iter = [50, 30, 15]
    current_transformation = np.identity(4)
    source = clean_pcd(source)
    target = clean_pcd(target)
    fitness = 0
    inlier_rmse = 0
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        if result_icp.fitness != 0:
            current_transformation = result_icp.transformation
            fitness = result_icp.fitness
            inlier_rmse = result_icp.inlier_rmse
        else:
            break

    merge = source.transform(current_transformation) + target

    return merge, current_transformation, fitness, inlier_rmse


def rgbd2pcd(color, depth, K, dist=None, bbox=None, depth_scale=4, depth_trunc=2000):
    """
    generate point cloud from an rgbd image
    :param color:
    :param depth:
    :param K:
    :param dist:
    :param bbox:
    :param depth_scale:
    :param depth_trunc:
    :return:
    """
    if bbox is not None:
        depth_new = -np.ones(np.shape(depth), dtype=depth.dtype)
        x, y, w, h = bbox
        depth_new[y:y + h, x:x + w] = depth[y:y + h, x:x + w]
    else:
        depth_new = depth

    if dist is not None:
        color = cv2.undistort(color, K, dist)
        depth_new = undistort_depth(depth_new, K, dist)

    height, width, _ = np.shape(color)

    img_color = o3d.geometry.Image(color)
    img_depth = o3d.geometry.Image(depth_new)
    fx, fy, ppx, ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color,
                                                              img_depth,
                                                              depth_scale=depth_scale,
                                                              depth_trunc=depth_trunc,
                                                              convert_rgb_to_intensity=False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    return pcd


def generate_square_pts(pattern=(9, 6), square=25):
    """
    Generate 3d positions of calibration patterns
    :param pattern:
    :param square:
    :return:
    """
    corners = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    # Corners is now a list of cooridnates of squares with side length of 25.
    corners[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    return corners.tolist()


def detect_subpix_chessboard(image, chessboard_pattern):
    """
    Detect chessboard corners for an image
    :param gray:
    :param chessboard_pattern:
    :return:
    """

    # Covert image to gray scale
    if len(np.shape(image)) == 3:
        # color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(np.shape(image)) == 2:
        # gray
        gray = image
    else:
        print("Image format incorrect!")
        sys.exit(1)

    # termination criteria
    flags = cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners_2d = cv2.findChessboardCorners(gray, chessboard_pattern, None, flags)
    if not ret:
        return None
    cv2.cornerSubPix(gray, corners_2d, (11, 11), (-1, -1), criteria)  # subpixel
    corners_2d.resize((corners_2d.shape[0], corners_2d.shape[2]))
    return corners_2d


def Rt2T(R, t):
    """
    Convert R (3x3),t (3x1) to T (4x4)
    :param R:
    :param t:
    :return:
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.squeeze(t)
    return T


def rename_rgb_files(folder_path, save_path, num_cam, start_num):
    """
    Rename rgb files from folder_path in sequential order in save_path
    Args:
        folder_path: Path of folder that stores rgb files
        save_path: Path of folder that stores renamed rgb files
        num_cam: Number of cameras that captured rgb images
        start_num: The index that the renamed sequence starts

    Returns:
        The index of the upcoming rgb image to be renamed to.
    """
    for i in range(num_cam):
        index = start_num
        rgb_files = list(natsorted(glob.glob(os.path.join(folder_path, "*_%d_rgb.png" % i))))
        for file in rgb_files:
            os.rename(file, file[:-4] + "_old" + file[-4:])
        for file in rgb_files:
            os.rename(file[:-4] + "_old" + file[-4:], os.path.join(save_path, "%d_%d_rgb.png" % (index, i)))
            index += 1
    return index


def rename_depth_files(folder_path, save_path, num_cam, start_num):
    """
    Rename depth files from folder_path in sequential order in save_path
    Args:
        folder_path: Path of folder that stores depth files
        save_path: Path of folder that stores renamed depth files
        num_cam: Number of cameras that captured depth images
        start_num: The index that the renamed sequence starts

    Returns:
        The index of the upcoming depth image to be renamed to.

    """
    for i in range(num_cam):
        index = start_num
        depth_files = list(natsorted(glob.glob(os.path.join(folder_path, "*_%d_depth.png" % i))))
        for file in depth_files:
            os.rename(file, file[:-4] + "_old" + file[-4:])
        for file in depth_files:
            os.rename(file[:-4] + "_old" + file[-4:], os.path.join(save_path, "%d_%d_depth.png" % (index, i)))
            index += 1
    return index


def rename_all_files(folder_path, save_path, num_cam, start_num):
    """
    Rename both rgb and depth image in sequential order
    Args:
        folder_path: Path of folder that stores rgb and depth files
        save_path: Path of folder that stores renamed rgb and depth files
        num_cam: Number of cameras that captured rgb and depth images
        start_num: The index that the renamed sequence starts

    Returns:
        The index of the upcoming depth image to be renamed to.
    """
    index_rgb = rename_rgb_files(folder_path, save_path, num_cam, start_num)
    index_depth = rename_depth_files(folder_path, save_path, num_cam, start_num)
    assert (index_rgb == index_depth)
    return index_depth


def produce_pcd_from_image_pairs(record_path, save_path, cam0_intrin, cam1_intrin):
    """
    Produces pointclouds from RGBD images collected by cam0 and cam1 and save them.
    Args:
        record_path: Path to the folder that stores RGBD images
        save_path: Path to the folder that saves produced pointclouds
        cam0_intrin: Path to the file that stores intrinsic parameters of camera 0
        cam1_intrin: Path to the file that stores instrinsic parameters of camera 1

    Returns: None

    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num_cam = 2
    num_files = len(list(natsorted(glob.glob(os.path.join(record_path, "*_0_rgb.png")))))
    with open(cam0_intrin, "r") as intrin:
        intrinsic = json.load(intrin)
    K_0 = np.array(intrinsic['K'])
    dis_0 = np.array(intrinsic['dist'])
    with open(cam1_intrin, "r") as intrin:
        intrinsic = json.load(intrin)
    K_1 = np.array(intrinsic['K'])
    dis_1 = np.array(intrinsic['dist'])

    for i in range(num_files):
        for j in range(num_cam):
            rgb_file = cv2.imread(os.path.join(record_path, "%d_%d_rgb.png" % (i, j)))
            depth_file = cv2.imread(os.path.join(record_path, "%d_%d_depth.png" % (i, j)), -1)
            rgb_file = cv2.cvtColor(rgb_file, cv2.COLOR_BGR2RGB)
            if j == 0:
                K = K_0
                dis = dis_0
            else:
                K = K_1
                dis = dis_1
            pcd = rgbd2pcd(rgb_file, depth_file, K, dis)
            o3d.io.write_point_cloud(os.path.join(save_path, "%d_%d.ply" % (i, j)), pcd)


def visualise_calibration_results(target, source, trans_matrix_path):
    f = open(trans_matrix_path)
    trans_matrix = json.load(f)
    merge, current_transformation, fitness, inlier_rmse = color_icp(source, target, np.array(trans_matrix['R']))
    o3d.visualization.draw_geometries([merge])
    print("The current transformation matrix is \n")
    print(current_transformation)
    print("Fitness: %f \n" % fitness)
    print("Inlier RMSE: %f " % inlier_rmse)


def load_sorted_files(file_path, intrinsics=False):
    """
    Load sorted rgb and depth file names saved in the file path
    Args:
        file_path: Path to the image folder that stores files of sorted image names

    Returns:

    """
    cam0_rgb = np.load(os.path.join(file_path, "cam0_rgb.npy"))
    cam0_depth = np.load(os.path.join(file_path, "cam0_depth.npy"))
    cam1_rgb = np.load(os.path.join(file_path, "cam1_rgb.npy"))
    cam1_depth = np.load(os.path.join(file_path, "cam1_depth.npy"))
    if not intrinsics:
        return cam0_rgb, cam0_depth, cam1_rgb, cam1_depth
    else:
        cam0_intrinsics = json.load(open(os.path.join(file_path, "info_0.json")))
        cam1_intrinsics = json.load(open(os.path.join(file_path, "info_1.json")))
        return cam0_rgb, cam0_depth, cam1_rgb, cam1_depth, cam0_intrinsics, cam1_intrinsics


def view_pointcloud(pcd, view='top_view'):
    if view == 'top_view':
        front = [-0.62897039144127798, 0.0085139465003027, -0.77738263384590445]
        lookat = [593.7023553341794, -661.75659260275165, 1221.9251928709737]
        up = [0.77742850114894091, 0.0082807203569712062, -0.6289168110900526]
        zoom = 0.63500000000000012
    elif view == "left_view":
        front = [-0.88850862739825065, 0.033800211880633542, 0.4576133353779161]
        lookat = [336.20103389354148, -687.56374102482846, 467.50026367810102]
        up = [-0.45581064965760909, 0.049766759534306884, -0.88868437665132982]
        zoom = 0.43499999999999994
    elif view == "right_view":
        front = [0.32040885855293533, -0.00076691139859819232, -0.94727903766931953]
        lookat = [154.10127362115884, -700.78548432304694, 460.86444005963972]
        up = [-0.94658340664687979, 0.03806581016446204, -0.32020438528744588]
        zoom = 0.29499999999999982

    print("Press q to close the window")
    o3d.visualization.draw_geometries([pcd], front=front, lookat=lookat, up=up, zoom=zoom)


def load_intrinsics(folderpath, filename):
    intrinsics = json.load(open(os.path.join(folderpath, filename)))
    K = np.array(intrinsics['K'])
    dist = np.array(intrinsics['dist'])
    return K, dist

