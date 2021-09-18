import os

import configuration as config

import flow_vis
import flow_warp
import spherical_coordinates

from utility import projection
from utility import image_io
from utility import mocker_data_generator as MDG


import numpy as np


def test_get_blend_weight_ico():
    """"""
    height = 50
    width = 40
    x_list = np.linspace(-1.0, 1.0, width, endpoint=True)
    y_list = np.linspace(-1.0, 1.0, height, endpoint=True)
    face_x_src_gnomonic, face_y_src_gnomonic = np.meshgrid(x_list, y_list)
    face_x_src_gnomonic = face_x_src_gnomonic.reshape((-1))
    face_y_src_gnomonic = face_y_src_gnomonic.reshape((-1))

    # [h,w,2]
    flow_uv = MDG.opticalflow_random(height, width)

    gnomonic_bounding_box = np.array([[-0.8, -0.8], [0, 0.8], [0.8, -0.8]], dtype=np.float64)

    # straightforward, cartesian_distance_log, cartesian_distance_exp, normal_distribution, normal_distribution_flowcenter, image_warp_error
    weight_list = ["image_warp_error"]
    print("Weight List: {}".format(weight_list))
    if "straightforward" in weight_list:
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "straightforward",
                                                     gnomonic_bounding_box=gnomonic_bounding_box)
    elif "cartesian_distance_log" in weight_list:
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "cartesian_distance_log")
    elif "cartesian_distance_exp" in weight_list:
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "cartesian_distance_exp")
    elif "normal_distribution" in weight_list:
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "normal_distribution")
    elif "normal_distribution_flowcenter" in weight_list:
        # flow_vis.flow_value_to_color(flow_uv)
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "normal_distribution_flowcenter", flow_uv)
    elif "image_warp_error" in weight_list:
        image_erp_src = MDG.image_square(height, width)
        image_erp_tar = MDG.image_square(height, width)
        weight_mat = projection.get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic, "image_warp_error", flow_uv,
                                                     image_erp_src=image_erp_src, image_erp_tar=image_erp_tar)

    weight_mat = weight_mat.reshape((height, width))
    image_io.image_show(weight_mat)


def test_get_blend_weight_cubemap():
    pass


def test_flow_rotate_endpoint(erp_src_image_filepath, erp_tar_image_filepath):
    """ 
    Test warp optical flow. Always rotate the target image.
    """
    src_image_data = image_io.image_read(erp_src_image_filepath)
    # tar_image_data = image_io.image_read(erp_tar_image_filepath)

    # 0) rotation tar image
    rotation_theta = np.radians(30.0)
    rotation_phi = np.radians(30.0)

    # 1) compute the flow from src to rotated rotated tar image
    rotation_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi, False)
    tar_image_data_rot = spherical_coordinates.rotate_erp_array(src_image_data, rotation_mat)
    image_io.image_save(tar_image_data_rot, erp_tar_image_filepath + "_rot.jpg")
    # # genera
    # flow_dis = flow_estimate.of_methdod_DIS(src_image_data, tar_image_data_rot)
    rotation_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi, False)
    flow_dis = spherical_coordinates.rotation2erp_motion_vector(src_image_data.shape[0:2], rotation_mat)
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


def test_get_padding_vs_fov_plot():
    """ Plot the relationship between the padding and FoV. """
    projection.get_padding_vs_fov_plot()


if __name__ == "__main__":
    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")

    erp_flow_gt_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    erp_flow_dis_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/cubemap_flo_dis_padding_stitch.flo")

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')

    args = parser.parse_args()

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")

    test_list = []
    test_list.append(args.task)

    if 1 in test_list:
        # test_get_rotation(erp_src_image_filepath, erp_flow_gt_filepath)
        # test_get_rotation(erp_src_image_filepath, erp_flow_dis_filepath)
        test_get_blend_weight_ico()
    if 2 in test_list:
        # test_flow_accumulate_endpoint(erp_src_image_filepath, erp_tar_image_filepath)
        test_get_blend_weight_cubemap()
    if 3 in test_list:
        # test_flow2rotation_2d(erp_src_image_filepath, erp_tar_image_filepath)
        test_flow_rotate_endpoint()
    if 4 in test_list:
        test_get_padding_vs_fov_plot()
