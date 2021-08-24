import configuration as config


from utility import image_io
from utility import flow_io
from utility import flow_vis

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
            of_data_vis = flow_vis.flow_to_color(of_data, min_ratio=0.2, max_ratio=0.8)  # ,  min_ratio=0.3, max_ratio=0.97)
            # of_data_vis = flow_vis.flow_value_to_color(of_data, min_ratio=0.2, max_ratio=0.8)
            image_io.image_save(of_data_vis, data_dir + filename + ".jpg")
            # print("visual optical flow {}".format(filename))
            # of_data_vis_uv = flow_vis.flow_max_min_visual(of_data, None)#"D:/1.jpg")


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
    uv_geo = flow_vis.flow_pix2geo(flo_data[:,:,0], flo_data[:,:,1])
    image_io.image_show(uv_geo)


def test_flow_uv_to_colors():
    image_width = 80
    image_height = 40
    u = np.full((image_width, image_height), 3, dtype=np.float64)
    v = np.full((image_width, image_height), 0, dtype=np.float64)

    flow_vis.flow_uv_to_colors(u, v, )


if __name__ == "__main__":
    # vis_of_folder("D:/workdata/omniphoto_bmvc_2021/BathAbbey2/result/pwcnet/")
    # of_data = flow_io.readFlowFile("D:/workdata/omniphoto_bmvc_2021/BathAbbey2/result/pwcnet/")
    # image_io.image_show(of_data[:,:,0])
    # image_io.image_show(of_data[:,:,1])
    # of_data_vis = flow_vis.flow_to_color(of_data, [-3, 3])
    # image_io.image_show(of_data_vis)
    # test_flow_uv_to_colors()
    # test_create_colorwheel_bar()
    # test_flow_pix2geo()
    test_flow_pix2geo_0()
