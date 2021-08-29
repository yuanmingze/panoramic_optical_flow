import os

import configuration as config
import flow_estimate
import flow_vis
import flow_warp
import spherical_coordinates

from utility import projection
from utility import image_io
from utility import flow_io
from utility import flow_postproc

import numpy as np


def test_get_rotation(erp_src_image_path, erp_flow_path):
    """Test the get_rotation function.
    """
    erp_image = image_io.image_read(erp_src_image_path)
    erp_flow = flow_io.flow_read(erp_flow_path)

    erp_flow = flow_postproc.erp_of_wraparound(erp_flow)

    # get the rotation
    erp_image_rotated = flow_warp.global_rotation_warping(erp_image, erp_flow)

    # rotate the src image
    rotated_image_path = erp_src_image_path + "_rotated.jpg"
    print("output rotated image: {}".format(rotated_image_path))
    image_io.image_save(erp_image_rotated, rotated_image_path)


def test_flow_accumulate_endpoint(erp_src_image_filepath, erp_tar_image_filepath):
    """ 
    Test warp optical flow. Always rotate the target image.
    """
    src_image_data = image_io.image_read(erp_src_image_filepath)
    # tar_image_data = image_io.image_read(erp_tar_image_filepath)

    # 0) rotation tar image
    rotation_theta = np.radians(30.0)
    rotation_phi = np.radians(30.0)

    # 1) compute the flow from src to rotated rotated tar image
    tar_image_data_rot, rotation_mat = spherical_coordinates.rotate_erp_array(src_image_data, rotation_theta, rotation_phi)
    image_io.image_save(tar_image_data_rot, erp_tar_image_filepath + "_rot.jpg")
    # # genera
    # flow_dis = flow_estimate.of_methdod_DIS(src_image_data, tar_image_data_rot)
    flow_dis, rotation_mat = spherical_coordinates.rotation2erp_motion_vector(src_image_data.shape[0:2], rotation_theta, rotation_phi)
    flow_vis_data = flow_vis.flow_to_color(flow_dis, min_ratio=0.2, max_ratio=0.8)
    image_io.image_save(flow_vis_data, erp_src_image_filepath + "_flow.jpg")
    tar_image_data_warp = flow_warp.warp_backward(tar_image_data_rot, flow_dis)
    image_io.image_save(tar_image_data_warp, erp_tar_image_filepath + "_rot_flow.jpg")

    # 2) get the flow from src to tar image
    # warp the optical flow base on rotation
    flow_dis_rot = projection.flow_rotate_endpoint(flow_dis, rotation_mat.T)
    flow_vis_data = flow_vis.flow_to_color(flow_dis_rot, min_ratio=0.2, max_ratio=0.8)
    image_io.image_save(flow_vis_data, erp_src_image_filepath + "_flow_rot.jpg")
    src_image_data_rot = flow_warp.warp_forward(src_image_data, flow_dis_rot, True)
    image_io.image_save(src_image_data_rot, erp_src_image_filepath + "_warp_forward_rot.jpg")
    src_image_data_rot = flow_warp.warp_backward(src_image_data, flow_dis_rot)
    image_io.image_save(src_image_data_rot, erp_src_image_filepath + "_warp_backward_rot.jpg")


def test_flow2rotation_2d(erp_src_image_filepath, erp_tar_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)
    tar_image_data = image_io.image_read(erp_tar_image_filepath)
    # flow_array = flow_estimate.of_methdod_DIS(src_image_data, tar_image_data)
    
    # theta, phi
    rotation_list = [(-20, 10.0), (30.0, -10.0), (30.0, 15.0), (-30.0, 15.0), (-25.0, -13.0)]
    for rotation in rotation_list:
        rotation_theta = np.radians(rotation[0])
        rotation_phi = np.radians(rotation[1])
        flow_array, _ = spherical_coordinates.rotation2erp_motion_vector(src_image_data.shape[0:2], rotation_theta, rotation_phi)
        # flow_vis.flow_value_to_color(flow_array)
        theta_delta, phi_delta = projection.flow2rotation_2d(flow_array, False)
        print("original rotation: {},{}, result is: {}, {}".
              format(np.degrees(rotation_theta), np.degrees(rotation_phi), np.degrees(theta_delta), np.degrees(phi_delta)))


if __name__ == "__main__":
    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")

    erp_flow_gt_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    erp_flow_dis_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/cubemap_flo_dis_padding_stitch.flo")

    # test_get_rotation(erp_src_image_filepath, erp_flow_gt_filepath)
    # test_get_rotation(erp_src_image_filepath, erp_flow_dis_filepath)
    # test_flow_accumulate_endpoint(erp_src_image_filepath, erp_tar_image_filepath)
    test_flow2rotation_2d(erp_src_image_filepath, erp_tar_image_filepath)
