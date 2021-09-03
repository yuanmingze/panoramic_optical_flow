import configuration as config
import flow_postproc

from utility import flow_estimate
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import flow_warp

import os

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False

def test_DIS_flow(erp_src_image_filepath, erp_tar_image_filepath, erp_of_dif_output_filepath):
    # load image to CPU memory
    src_erp_image = image_io.image_read(erp_src_image_filepath)
    tar_erp_image = image_io.image_read(erp_tar_image_filepath)

    of_forward = flow_estimate.of_methdod_DIS(src_erp_image, tar_erp_image)
    flow_io.flow_write(of_forward, erp_of_dif_output_filepath)
    of_forward_vis = flow_vis.flow_to_color(of_forward)
    image_io.image_save(of_forward_vis, erp_of_dif_output_filepath + ".jpg")
    # flow_vis.flow_value_to_color(of_forward, visual_colormap="jet")


def test_pano_of_our(erp_src_image_filepath, erp_tar_image_filepath, erp_opticalflow_debug_dir):
    """Test panoramic optical flow method 0.
    """
    # load image to CPU memory
    src_erp_image = image_io.image_read(erp_src_image_filepath)
    tar_erp_image = image_io.image_read(erp_tar_image_filepath)

    print("compute optical flow by multi-step DIS")
    flow_estimator = flow_estimate.PanoOpticalFlow()
    flow_estimator.debug_output_dir = erp_opticalflow_debug_dir
    log.info("debug output folder: {}".format(erp_opticalflow_debug_dir))
    flow_estimator.debug_enable = True
    # flow_estimator.erp_enable = False
    # flow_estimator.cubemap_enable = False
    # flow_estimator.ico_enable = False
    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
    optical_flow = flow_postproc.erp_of_wraparound(optical_flow)

    # output optical flow
    flow_io.flow_write(optical_flow, erp_opticalflow_debug_dir + "pano_of_0_final_visul.flo")
    optical_flow_vis = flow_vis.flow_to_color(optical_flow)
    image_io.image_save(optical_flow_vis, erp_opticalflow_debug_dir + "pano_of_0_final_visul.jpg")

    # warp the source image with optical flow
    src_erp_image_warp = flow_warp.warp_forward(src_erp_image, optical_flow, wrap_around=True)
    image_io.image_save(src_erp_image_warp, erp_src_image_filepath + "_multi_step_dis_warp.jpg")


if __name__ == "__main__":
    padding_size = 0.1
    ico_face_number = 20

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")

    erp_opticalflow_output_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_multisteps.flo")
    erp_of_dif_output_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward_dis.flo")

    erp_opticalflow_debug_output_dir = config.TEST_data_root_dir + "replica_360/apartment_0/debug/"

    test_list = [0]
    if 0 in test_list:
        # test_DIS_flow(erp_src_image_filepath, erp_tar_image_filepath, erp_of_dif_output_filepath)
        test_pano_of_our(erp_src_image_filepath, erp_tar_image_filepath, erp_opticalflow_debug_output_dir)

    if 1 in test_list:
        # test perspective optical flow wrap-around
        data_root = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0_0/")
        erp_src_image_filepath = data_root + "0000_B_rgb.jpg"
        erp_tar_image_filepath = data_root + "0001_B_rgb.jpg"
        erp_opticalflow_output_filepath = data_root + "0001_rgb_multisteps.flo"
        test_DIS_flow(erp_src_image_filepath, erp_tar_image_filepath, erp_opticalflow_output_filepath)
