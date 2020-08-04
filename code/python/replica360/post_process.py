import math
import csv
import re
import sys
import pathlib
from datetime import datetime
import os
import shutil
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import msgpack


# the filename with re expression
# --- abbreviation ---
# fw = forward, bw = backward, vis = visual, 
# fne = filename expression or full name, pc = point cloud
# mp = MegaParallax
# prep = preprocessing
# mfos = modelFiles.openvslam
# ov = openvslam
FPS = 1
image_width = 1920
image_height = 960

index_digit_re = r"\d{4}"
index_digit_format = r"{:04d}"

# render step filename
rgb_image_fne = index_digit_re + r"_rgb.jpg"
depth_bin_fne = index_digit_re + r"_depth.bin"
flow_fw_bin_fne = index_digit_re + r"_opticalflow_forward.bin"
flow_bw_bin_fne = index_digit_re + r"_opticalflow_forward.bin"
center_rgb_fne = r"centre_rgb.jpg"
center_depth_bin_fne = r"cnetre_depth.bin"

# convtion step filename
flow_fw_flo_fne = index_digit_re + r"_opticalflow_forward.flo"
flow_bw_flo_fne = index_digit_re + r"_opticalflow_backward.flo"
flow_fw_vis_fne = index_digit_re + r"_opticalflow_forward_visual.jpg"
flow_bw_vis_fne = index_digit_re + r"_opticalflow_backward_visual.jpg"
center_obj_fne = r"0000_depth.obj"
center_depth_vis_fne = r"0000_depth.jpg"

# MegaParallax preprocessing step filename
mp_prep_traj_csv_fne = r"frame_trajectory_with_filename.txt"
mp_prep_ov_traj_csv_fne = r"frame_trajectory.txt"
mp_prep_ov_traj_obj_fne = r"frame_trajectory.obj"
mp_prep_msg_fne = r"map.msg"
mp_prep_camera_txt_fne = r"cameras.txt"
mp_prep_rgb_fne = r"panoramic-"+ index_digit_re + r".jpg"
mp_prep_mfos_fne = r"modelFiles.openvslam"

# MegaParallax viewer step filename
mp_view_camera_csv_fne = r"Camera.csv"
mp_view_flow_fw_flo_fne = r"panoramic-" + index_digit_re + r"-FlowToNext.flo"
mp_view_flow_bw_flo_fne = r"panoramic-"+ index_digit_re + r"-FlowToPrevious.flo"
mp_view_pc_csv_fne = r"PointCloud.csv"
mp_view_centre_obj_fne = r"spherefit-depth-ground-truth.obj"
mp_view_json_fne = r"PreprocessingSetup-"+ index_digit_re + r".json"

def load_motion_vector_raw_data(binary_file_path, height, width):
    """
    load motion vector form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(height, width, 4)
    # imgplot = plt.imshow(data[:,:,0])
    # plt.show()
    return data


def load_depth_raw_data(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(height,width,1)
    # imgplot = plt.imshow(data[:,:,0])
    # plt.show()
    return data


def visual_depth_map(binary_file_path, image_width, image_height):
    """
    """
    binary_file_path = '/mnt/sda1/workspace_linux/replica360/data/test_00/0000_depth.bin'
    data = load_depth_raw_data(binary_file_path,image_height,image_width)
    plt.imshow(data)
    plt.show()


def load_pointcloud_from_obj(obj_file_path):
    """
    """
    point_cloud = []
    obj_file_handle = open(obj_file_path, 'r')
    for line in obj_file_handle:
        split = line.split()

        #if blank line, skip
        if not len(split):
            continue
        if split[0] == "v":
            point_cloud.append([float(split[1]), float(split[2]), float(split[3])])

    obj_file_handle.close()
    return point_cloud


def load_replica_camera_traj(traj_file_path):
    """
    the format:
    index 
    """
    camera_traj = []
    traj_file_handle = open(traj_file_path, 'r')
    for line in traj_file_handle:
        split = line.split()
        #if blank line, skip
        if not len(split):
            continue
        camera_traj.append(split)
    traj_file_handle.close()
    return camera_traj


def generate_mp_preprocess_files(replica_camera_traj_file, replica_camera_center_file, \
            original_traj_dir,  original_center_dir, dest_data_dir):
    """
    :param replica_camera_traj_file: the camera trajectory file 
    :param replica_camera_center_file:
    :param original_traj_dir:
    :param original_center_dir:
    :param dest_data_dir:
    """
    print("---Generate MegaParallax Preprocessing Necessary Files---")
    # 0) generate frame_trajectory_with_filename.csv files
    print("Generate {}..........".format(mp_prep_traj_csv_fne))
    replica_camera_poses = load_replica_camera_traj(replica_camera_traj_file)
    output_csv_header = ["#index", "filename", "trans_wc.x", "trans_wc.y", "trans_wc.z", "quat_wc.x", "quat_wc.y", "quat_wc.z", "quat_wc.w"]
 
    mp_prep_ov_traj_csv_han = open(dest_data_dir + mp_prep_ov_traj_csv_fne, 'w', newline='')
    openvslam_ov_traj_csv = csv.writer(mp_prep_ov_traj_csv_han, delimiter=' ', quoting=csv.QUOTE_MINIMAL)

    mp_prep_ov_traj_obj_han = open(dest_data_dir + mp_prep_ov_traj_obj_fne, 'w', newline='')

    # replica coordinate system to openvslam coordinate system
    coord_convert_r2o_R = R.from_matrix([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    with open(dest_data_dir + mp_prep_traj_csv_fne, 'w', newline='') as file_handle_output:
        openvslam_traj_csv = csv.writer(file_handle_output, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        openvslam_traj_csv.writerow(output_csv_header)
        for pose_item in replica_camera_poses:
            # image_idx = int(FPS * float(pose_item[0]) + 0.5)
            image_idx = int(pose_item[0])
            image_name = mp_prep_rgb_fne.replace(index_digit_re, index_digit_format.format(image_idx))

            
            # change rotation to quaternion
            camera_rotation_x = float(pose_item[4]) #/ 180.0 * math.pi
            camera_rotation_y = float(pose_item[5]) #/ 180.0 * math.pi
            camera_rotation_z = float(pose_item[6]) #/ 180.0 * math.pi
            
            # coordinate system conversion matrix from replica to openvslam
            # TODO: should be = coord_convert_r2o_R * rotation * coord_convert_r2o_R.inv()
            #rotation = R.from_euler('xyz', [camera_rotation_x, camera_rotation_y, camera_rotation_z], degrees=True)
            #rotation = coord_convert_r2o_R.inv() * rotation #* coord_convert_r2o_R.inv()
            #rotation = rotation * coord_convert_r2o_R

            # It's right
            rotation = R.from_euler('zxy', [camera_rotation_x, -camera_rotation_y, -camera_rotation_z], degrees=True)
            translation = coord_convert_r2o_R.apply(np.array([float(pose_item[1]), float(pose_item[2]), float(pose_item[3])]))

            mp_prep_ov_traj_obj_han.write("v {} {} {}\n".format(translation[0], translation[1], translation[2]))

            openvslam_traj_csv.writerow([image_idx, image_name, ] + list(translation) + rotation.as_quat().tolist())
            openvslam_ov_traj_csv.writerow([float(pose_item[0]) / 50.0] + list(translation) + rotation.as_quat().tolist())

    mp_prep_ov_traj_csv_han.close()
    mp_prep_ov_traj_obj_han.close()
    print("finished\n")

    # 1) generate map.msg file
    # load camera center 
    replica_center_camera_poses = load_replica_camera_traj(replica_camera_center_file)[0]
    replica_center_camera_pose = np.array([float(replica_center_camera_poses[1]), float(replica_center_camera_poses[2]), float(replica_center_camera_poses[3])])
    #coord_convert_r2o_R = R.from_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) # rotated in data format convertion
    print("Generate {}..........".format(mp_prep_msg_fne))
    point_cloud_data = load_pointcloud_from_obj(original_center_dir + center_obj_fne)
    landmarks = {}
    obj_points = []
    for idx in range(len(point_cloud_data)):
        point_word = replica_center_camera_pose  + np.array(point_cloud_data[idx])
        point = coord_convert_r2o_R.apply(point_word)
		# output for msg
        landmarks[str(idx)] = {"pos_w": list(point)}
		# output for obj
        obj_points.append(" ".join(["v",str(point[0]) ,str(point[1]) ,str(point[2])]))
    data_text = {"landmarks": landmarks}
    # pack and output
    msgpack_packer = msgpack.Packer(use_bin_type=True)
    mapmsg_file = open(dest_data_dir + mp_prep_msg_fne, 'wb')
    mapmsg_file.write(msgpack_packer.pack(data_text))
    mapmsg_file.close()
    # write of obj
    obj_file_handle = open(dest_data_dir + mp_prep_msg_fne + ".obj",'w')
    obj_points_str = '\n'.join(obj_points)
    obj_file_handle.write(obj_points_str)
    obj_file_handle.close()
    print("finished\n")

    # 2) cameras.txt
    print("Generate {}..........".format(mp_prep_camera_txt_fne))
    with open(dest_data_dir + mp_prep_camera_txt_fne , "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write("1 EQUIRECTANGULAR {} {} 500 {} {}".format(image_width, image_height, image_width / 2.0, image_height / 2.0))
    print("finished\n")

    # 3) modelFiles.openvslam
    print("Generate {}..........".format(mp_prep_mfos_fne))
    with open(dest_data_dir + mp_prep_mfos_fne , "w") as f:
        f.write("../Capture/openvslam/cameras.txt\n")
        f.write("../Capture/openvslam/frame_trajectory_with_filename.txt\n")
        f.write("../Capture/openvslam/map.msg\n")
    print("finished\n")

    # 4) copy and rename all rgb image filename regular expression
    print("Generate {}..........".format("rgb images"))
    original_traj_dir_path = pathlib.Path(original_traj_dir)
    rgb_image_fne_pattern = re.compile(rgb_image_fne)
    for item in original_traj_dir_path.iterdir():
        if not rgb_image_fne_pattern.match(item.name):
            continue
        # get index, new file name
        index_digt = re.search(index_digit_re, item.name)
        new_rgb_filename = mp_prep_rgb_fne.replace(index_digit_re,index_digt.group())
        new_rgb_file_path = dest_data_dir + new_rgb_filename
        shutil.copy(str(item), new_rgb_file_path)
        print("copy {} to {}".format(item.name, new_rgb_file_path))
    print("finished\n")
    print("---Finish Generate MegaParallax Preprocessing Necessary Files---")


def generate_mp_viewer_files(original_traj_dir, original_center_dir, dest_data_dir):
    """
    """
    # 1) generate camera files
    print("Generate {}..........".format(mp_view_camera_csv_fne))
    print("finished\n")

    # 2) generate the json file
    print("Generate {}..........".format(mp_view_json_fne))
    print("finished\n")

    # 3) genreate PointCloud.csv file
    print("Generate {}..........".format(mp_view_pc_csv_fne))
    print("finished\n")

    # 3)  copy and rename all flo files
    print("copy and rename the flo files.........")
    flow_fw_flo_fne_pattern = re.compile(flow_fw_flo_fne)
    flow_bw_flo_fne_pattern = re.compile(flow_bw_flo_fne)
    for item in pathlib.Path(original_traj_dir).iterdir():
        if flow_fw_flo_fne_pattern.match(item.name) is None and \
            flow_bw_flo_fne_pattern.match(item.name) is None:
            continue

        src_filename = item.name
        tar_filename = None
        index_digt = re.search(index_digit_re, item.name).group()

        if flow_fw_flo_fne_pattern.match(item.name):
            tar_filename = mp_view_flow_fw_flo_fne.replace(index_digit_re, index_digt)
        elif flow_bw_flo_fne_pattern.match(item.name):
            tar_filename = mp_view_flow_bw_flo_fne.replace(index_digit_re, index_digt)
        
        # get index, new file name
        print("copy flo files from {} to {}".format(original_center_dir + src_filename, dest_data_dir + tar_filename))
        shutil.copy(original_traj_dir + src_filename, dest_data_dir + tar_filename)
    print("finished\n")

    # 4) rename mesh files
    print("copy and rename center view point mesh......")
    src_obj_path = original_center_dir + center_obj_fne
    tar_obj_path = dest_data_dir + mp_view_centre_obj_fne
    shutil.copy(src_obj_path, tar_obj_path)
    print("finished\n")


def post_process(root_dir, dataset_name, \
                replica_camera_file, replica_center_camera_file, \
                original_traj_dir, original_center_dir, \
                preprocessing_folder = "Preprocessing/", viewer_folder = "Viewer/", \
                input_image_width = 3840, input_image_height = 1920, input_FPS = 50):
    """
    will generate megaparallax ready files on the root_dir, 
    named with *_Preprocessing and *_Viewer
    """
    global FPS
    FPS = input_FPS
    global image_width
    image_width = input_image_width
    global image_height
    image_height = input_image_height

    # `$dataset_name$_YY-MM-DD-HH-MM-SS_Preprocessing`.
    # timestamp_str = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
    # mp_prep_dest_data_dir = root_dir + dataset_name + "_"+ timestamp_str + "_Preprocessing/"
    # mp_viewer_dest_data_dir = root_dir + dataset_name + "_" + timestamp_str + "_Viewer/"
    mp_prep_dest_data_dir = root_dir + preprocessing_folder
    mp_viewer_dest_data_dir = root_dir + viewer_folder

    if os.path.exists(mp_prep_dest_data_dir):
        print("the folder: {}, exist, and remove it.".format(mp_prep_dest_data_dir))
        temp_file = root_dir + "Preprocessing_temp/"
        shutil.move(mp_prep_dest_data_dir, temp_file)
        shutil.rmtree(temp_file, ignore_errors=False, onerror=None)
    os.mkdir(mp_prep_dest_data_dir)

    if os.path.exists(mp_viewer_dest_data_dir): 
        print("the folder: {}, exist, and remove it.".format(mp_viewer_dest_data_dir))
        temp_file = root_dir + "Viewer_temp/"
        shutil.move(mp_viewer_dest_data_dir, temp_file)
        shutil.rmtree(temp_file, ignore_errors=False, onerror=None)
    os.mkdir(mp_viewer_dest_data_dir)

    print("Target preprocessing directory: {}".format(mp_prep_dest_data_dir))
    print("Target viewer directory: {}".format(mp_viewer_dest_data_dir))
    
    # 0) generate the MegaParallax Preprocessing step files
    generate_mp_preprocess_files(replica_camera_file, replica_center_camera_file, original_traj_dir, original_center_dir, mp_prep_dest_data_dir)

    # 1) genreate the MegaParallax Viewer step files
    generate_mp_viewer_files(original_traj_dir, original_center_dir, mp_viewer_dest_data_dir)


if __name__ == "__main__":
    """
    # Example: 
    # hotel_0
    # 1K: -d "hotel_0" -r "d:/workspace_linux/replica360/data/" -c "hotel_0_2020_05_10_15_21_25_circle.csv" -ot "hotel_0_2020-05-10-15-43-57/" -oc "hotel_0_2020-05-10-15-54-44/""
    # 4K: -d "hotel_0" -r "d:/workspace_linux/replica360/data/" -c "hotel_0_2020_05_10_15_21_25_circle.csv" -ot "hotel_0_2020-05-14-09-23-53/" -oc "hotel_0_2020-05-14-09-31-20/""
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'dataset_name',type=str, help='the dataset name')
    parser.add_argument('-r', dest = 'root_dir',type=str, help='the root directory')
    parser.add_argument('-ct', dest = 'replica_camera_file', type=str, help='replica render input camera trajectory file')
    parser.add_argument('-cc', dest = 'replica_center_camera_file', type=str, help='replica render input camera trajectory file')
    parser.add_argument('-ot', dest = 'original_traj_dir', type=str, help='the rendered result')
    parser.add_argument('-oc', dest = 'original_center_dir', type=str, help='the rendered  of center viewpoint')

    parser.add_argument('-fps', dest = 'FPS', type=int, default=50, help='the data fps')
    parser.add_argument('-h', dest = 'height', type=int, default=1920, help='the data height')  
    parser.add_argument('-w', dest = 'width', type=int, default=3840, help='the data width')

    args = parser.parse_args()
    if len(sys.argv) < 4:
        parser.print_help()
        parser.exit()

    print(args)
    # create absolute path
    dataset_name = args.dataset_name
    root_dir = pathlib.Path(args.root_dir)
    replica_camera_file = str(root_dir / args.replica_camera_file)
    replica_center_camera_file = str(root_dir / args.replica_center_camera_file)
    original_traj_dir = str(root_dir / args.original_traj_dir)+ "/"
    original_center_dir = str(root_dir / args.original_center_dir) + "/"

    print("post-process dataset: {} in root directory: {}".format(dataset_name, root_dir))
    print("With camera pose file: {} ".format(replica_camera_file))
    print("Original data directory: {}".format(original_traj_dir))
    print("Original center data directory: {}".format(original_center_dir))

    post_process(str(root_dir) + "/", dataset_name, \
                replica_camera_file, replica_center_camera_file, \
                original_traj_dir, original_center_dir, \
                input_image_width = args.width, input_image_height = args.height, input_FPS = args.FPS)