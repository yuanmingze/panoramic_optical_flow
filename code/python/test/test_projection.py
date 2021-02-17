import os

import configuration as config
import flow_estimate
import flow_vis
import flow_warp
import spherical_coordinates

from utility import projection
from utility import image_io
from utility import flow_io
from utility import flow_post_proc

import numpy as np

def test_get_rotation(erp_src_image_path, erp_flow_path):
    """Test the get_rotation function.
    """
    erp_image = image_io.image_read(erp_src_image_path)
    erp_flow = flow_io.flow_read(erp_flow_path)

    erp_flow = flow_post_proc.of_nonerp2erp(erp_flow)

    # get the rotation
    erp_image_rotated = projection.image_align(erp_image, erp_flow)

    # rotate the src image
    rotated_image_path = erp_src_image_path + "_rotated.jpg"
    print("output rotated image: {}".format(rotated_image_path))
    image_io.image_save(erp_image_rotated, rotated_image_path)


def test_flow_accumulate_endpoint(erp_src_image_filepath, erp_tar_image_filepath):
    """ Always rotate the target image.

    TODO if set rotation_longitude and rotation_latitude to 0.0, there are offset between the input and output image. Check the reason! 
    """
    src_image_data = image_io.image_read(erp_src_image_filepath)
    tar_image_data = image_io.image_read(erp_tar_image_filepath)

    # 0) rotation tar image
    # Note: 
    rotation_longitude = np.radians(10.0)
    rotation_latitude = np.radians(10.0)
    tar_image_data_rot = spherical_coordinates.rotate_array(tar_image_data, rotation_longitude, rotation_latitude)
    image_io.image_save(tar_image_data_rot, erp_tar_image_filepath + "_rot.jpg") 

    # 1) compute the flow from src to rotated rotated tar image
    flow_dis = flow_estimate.DIS(src_image_data, tar_image_data_rot)
    flow_vis_data = flow_vis.flow_to_color(flow_dis)
    image_io.image_save(flow_vis_data, erp_src_image_filepath + "_flow.jpg")

    # 2) get the flow from src to original tar image
    # warp the optical flow base on rotation
    flow_dis_rot = projection.flow_accumulate_endpoint(flow_dis, [-rotation_longitude, -rotation_latitude])
    flow_vis_data = flow_vis.flow_to_color(flow_dis_rot)
    image_io.image_save(flow_vis_data, erp_src_image_filepath + "_flow_rot.jpg")
    src_image_data_rot = flow_warp.warp_forward(src_image_data, flow_dis_rot)
    image_io.image_save(src_image_data_rot, erp_src_image_filepath + "_warp_rot.jpg")


if __name__ == "__main__":
    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")

    erp_flow_gt_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    erp_flow_dis_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/cubemap_flo_dis_padding_stitch.flo")

    # test_get_rotation(erp_src_image_filepath, erp_flow_gt_filepath)
    # test_get_rotation(erp_src_image_filepath, erp_flow_dis_filepath)
    test_flow_accumulate_endpoint(erp_src_image_filepath, erp_tar_image_filepath)