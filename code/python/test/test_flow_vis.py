import configuration as config
import mocker_data_generator as MDG


from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import flow_postproc

import numpy as np
import os


def vis_of_folder(data_dir):
    counter = 0
    for filename in os.listdir(data_dir):
        counter = counter + 1
        if counter % 10 == 0:
            print(f"{counter} : {filename}")

        if filename.endswith(".flo"):  # and filename == "0002_R_motionvector_forward.flo":
            of_data = flow_io.read_flow_flo(data_dir + filename)
            # of_data = flow_postproc.erp_of_unwraparound(of_data)
            of_data = flow_postproc.erp_of_wraparound(of_data)
            of_data_vis = flow_vis.flow_to_color(of_data, min_ratio=0.0, max_ratio=0.7, add_bar=True, sph_of=False)  # ,  min_ratio=0.3, max_ratio=0.97)
            # image_io.image_show(of_data_vis)
            image_io.image_save(of_data_vis, data_dir + filename + ".jpg")
            # of_data_vis = flow_vis.flow_value_to_color(of_data, min_ratio=0.2, max_ratio=0.8)
            # print("visual optical flow {}".format(filename))
            # of_data_vis_uv = flow_vis.flow_max_min_visual(of_data, None)#"D:/1.jpg")


def test_flow_pix2angle():
    """ """
    erp_image_height = 50
    erp_opticalflow = MDG.opticalflow_simple(erp_image_height, erp_image_height * 2, 20, 10)

    # erp_opticalflow_uv = np.moveaxis(erp_opticalflow, 0, -1)
    # vis_of = flow_vis.flow_value_to_color(erp_opticalflow_uv)
    # image_io.image_show(vis_of)

    flow_angle = flow_vis.flow_pix2sphangle(erp_opticalflow[0, :, :], erp_opticalflow[1, :, :])
    image_io.image_show(flow_angle)


def test_flow_to_color():
    of_file_path = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    of_data = flow_io.read_flow_flo(of_file_path)
    flow_vis_sph = flow_vis.flow_to_color(of_data, sph_of=True)
    image_io.image_show(flow_vis_sph)

def test_create_colorwheel_bar():
    #
    color_wheel_image = flow_vis.create_colorwheel_bar(500)
    image_io.image_show(color_wheel_image)


def test_flow_pix2geo():
    image_height = 400
    image_width = image_height * 2
    u = np.full((image_height, image_width), 0, dtype=np.float64)
    v = np.full((image_height, image_width), 3, dtype=np.float64)
    uv_geo = flow_vis.flow_pix2geo(u, v)
    image_io.image_show(uv_geo)


def test_flow_pix2geo_0():
    # load *.flo file
    flo_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    flo_data = flow_io.read_flow_flo(flo_filepath)
    uv_geo = flow_vis.flow_pix2geo(flo_data[:, :, 0], flo_data[:, :, 1])
    image_io.image_show(uv_geo)


def test_flow_uv_to_colors():
    image_width = 80
    image_height = 40
    u = np.full((image_width, image_height), 3, dtype=np.float64)
    v = np.full((image_width, image_height), 0, dtype=np.float64)

    flow_vis.flow_uv_to_colors(u, v)


if __name__ == "__main__":

    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/"
    data_dir = root_dir + "replica_seq_data/"

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')

    args = parser.parse_args()

    # data_dir = "D:/workdata/omniphoto_bmvc_2021/BathAbbey2/result/pwcnet/"
    # data_dir = "D:/workdata/opticalflow_data_bmvc_2021/apartment_0_rand_1k_0/cubemap/"
    # data_dir = "/mnt/sda1/workspace_linux/OmniFlowNet/LiteFlowNet/models/testing/results/"
    # data_dir = "/mnt/sda1/workspace_linux/OmniFlowNet/LiteFlowNet2/models/testing/results/"
    # data_dir = "/mnt/sda1/workspace_linux/OmniFlowNet/OmniFlowNet/models/testing/results/"
    # data_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/hotel_0_line_1k_0/result/omniflownet/"
    # data_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/apartment_0_circ_1k_0/result/omniflownet/"
    # data_dir = "D:/workdata/opticalflow_data_bmvc_2021/apartment_0_rand_1k_0/result/raft/"
    # data_dir = config.TEST_data_root_dir + "/replica_360/apartment_0_line_1k_0/cubemap_flo/"
    # data_dir = config.TEST_data_root_dir + "/replica_360/office_0_line_cubemap_stitch_debug/cubemap_flo/"
    data_dir = config.TEST_data_root_dir + "replica_360_cubemap/office_0_circ_pano/"

    test_list = []
    test_list.append(args.task)

    if 0 in test_list:
        vis_of_folder(data_dir)
    if 1 in test_list:
        test_create_colorwheel_bar()
    if 2 in test_list:
        test_flow_pix2geo()
    if 3 in test_list:
        test_flow_pix2geo_0()
    if 4 in test_list:
        test_flow_uv_to_colors()
    if 5 in test_list:
        test_flow_pix2angle()
    if 6 in test_list:
        test_flow_to_color()
