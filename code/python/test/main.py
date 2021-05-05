import configuration as config

from utility import image_io
from utility import flow_estimate
from utility import image_io
from utility import flow_io
from utility import flow_vis

from utility import flow_warp

from pathlib import Path


def pano_optical_flow():
    """
    The whole pipeline of panoramic optical flow.
    """
    # 0) set up the variables
    data_root = config.TEST_data_root_dir + "replica_360/apartment_0/"

    debug_enable = True
    if debug_enable:
        debug_output_dir = data_root + "debug/"
        Path(debug_output_dir).mkdir(parents=True, exist_ok=True)

    erp_image_src_path = data_root + "0001_rgb.jpg"
    erp_image_tar_path = data_root + "0002_rgb.jpg"
    erp_opticalflow_path = data_root + "0001_rgb_fw.flo"

    erp_image_src = image_io.image_read(erp_image_src_path)
    erp_image_tar = image_io.image_read(erp_image_tar_path)

    # 1) compute the 360 optical flow
    print("compute optical flow by multi-step DIS")
    optical_flow = flow_estimate.pano_optical_flow(erp_image_src, erp_image_tar, debug_output_dir = debug_output_dir)

    if debug_enable:
        # warp the source image with optical flow
        src_erp_image_warp = flow_warp.warp_forward(erp_image_src, optical_flow)
        image_io.image_save(src_erp_image_warp, erp_opticalflow_path + "_multi_step_dis_warp.jpg")

    # 2) output to file and visualization
    flow_io.flow_write(optical_flow, erp_opticalflow_path)
    optical_flow_vis = flow_vis.flow_to_color(optical_flow)
    image_io.image_save(optical_flow_vis, erp_opticalflow_path + ".jpg")


if __name__ == "__main__":
    pano_optical_flow()
