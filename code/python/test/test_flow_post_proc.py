import configuration as config

import flow_postproc
import flow_warp
import image_utility

from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import flow_postproc


def test_flow_resize(data_dir):
    optical_flow_file = data_dir + "replica_360/hotel_0/0001_opticalflow_forward.flo"
    image_source = data_dir + "replica_360/hotel_0/0001_rgb.jpg"
    image_target = data_dir + "replica_360/hotel_0/0002_rgb.jpg"

    resize_ratio = 0.4

    of_data = flow_io.flow_read(optical_flow_file)
    height = of_data.shape[0]
    width = of_data.shape[1]
    of_data = flow_postproc.flow_resize(of_data, resize_ratio=resize_ratio)
    # of_data_resize_vis = flow_vis.flow_to_color(of_data_resize)
    # image_io.image_show(of_data_resize_vis)
    of_data = flow_postproc.flow_resize(of_data, width_new=width, height_new=height)

    # test warp the image
    image_src = image_io.image_read(image_source)
    image_tar = image_io.image_read(image_target)
    # image_src = image_utility.image_resize(image_src, image_ratio=resize_ratio)

    image_src_warped = flow_warp.warp_forward(image_src, of_data)
    image_io.image_show(image_src_warped)
    image_io.image_show(image_tar)
    image_io.image_diff(image_src_warped, image_tar)


def test_convert_warp_around():
    of_gt = flow_io.readFlowFile("../../data/replica_360/hotel_0/0001_opticalflow_forward.flo")
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)

    of_gt = flow_postproc.convert_warp_around(of_gt)
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)


if __name__ == "__main__":
    test_list = [0]

    data_dir = config.TEST_data_root_dir

    if 0 in test_list:
        test_flow_resize(data_dir)

    if 1 in test_list:
        test_convert_warp_around(data_dir)
