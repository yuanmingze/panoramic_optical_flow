import os

import configuration as config

from utility import projection
from utility import image_io
from utility import flow_io
from utility import flow_post_proc


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


if __name__ == "__main__":
    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")

    erp_flow_gt_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    erp_flow_dis_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/cubemap_flo_dis_padding_stitch.flo")

    # test_get_rotation(erp_src_image_filepath, erp_flow_gt_filepath)
    test_get_rotation(erp_src_image_filepath, erp_flow_dis_filepath)
