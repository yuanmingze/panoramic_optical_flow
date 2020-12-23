import os
import pathlib

import numpy as np
from PIL import Image

import png
import itertools

import configuration as config

from utility import depth_io
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import image_io
from utility import depth_io
from utility import flow_warp
from utility import gnomonic_projection as gp

path = os.getcwd()
TEST_DATA_ROOT_FOLDER = path + "/../../data/replica_360/office_0/"
TEMP_DATA_ROOT_FOLDER = path + "/../../data/test_temp/"


def of_vis(of_data_dir):
    """
    """
    of_data_path = pathlib.Path(of_data_dir)
    for file_path in of_data_path.iterdir():
        if  file_path.suffix == ".floss":
            of_data = flow_io.readFlowFloss(str(file_path))
        if  file_path.suffix == ".flo":
            of_data = flow_io.readFlowFile(str(file_path))
        else:
            continue

        print(file_path)
        of_data_color = flow_vis.flow_to_color(of_data)
        flow_visual_file_path = str(file_path) + ".jpg"
        image_io.image_save(of_data_color, flow_visual_file_path)


def depth_map_format_transfrom():
    """
    """
    root_dir = "D:/workdata/casual_stereo_vr_2020_test/central_viewpoint_rgbd/library/"
    depth_dpt_path = root_dir + "0000_depth.dpt"
    depth_png_path = root_dir + "0000_depth.png"
    depth_visual_path = root_dir + "0000_depth.jpg"

    depth_data = depth_io.read_dpt(depth_dpt_path)

    depth_visual = depth_io.depth_visual(depth_data)
    image_io.image_save(depth_visual, depth_visual_path)

    depth_io.write_png(depth_data, depth_png_path)


def test_depth_io():
    """
    test depht io
    """
    dpt_file_path = TEST_DATA_ROOT_FOLDER + "0000_depth.dpt"

    # load grount truth from bin file
    #depht_io.write_dpt(depth_data, dpt_file_path)
    depth_data = depth_io.read_dpt(dpt_file_path)

    image_io.image_show(depth_data)

    width = np.shape(depth_data)[0]
    height = np.shape(depth_data)[1]
    # test dpt format

    binary_file_path = TEMP_DATA_ROOT_FOLDER + "0000_depth.bin"
    png_file_path = TEMP_DATA_ROOT_FOLDER + "0000_depth.png"

    # depth_io.read_bin(depth_data, binary_file_path, height, width)
    # image_io.image_show(depth_data)

    # test png format
    depth_io.write_png(depth_data, png_file_path)
    depth_data = depth_io.read_png(png_file_path)
    image_io.image_show(depth_data)


def depth_visual():
    """
    """
    depth_file_path = "/mnt/sda1/workdata/Casual 3D Photography_Datasets/boatshed/stitch/back/disparities.png"
    # depth_file_path = "/mnt/sda1/workdata/Casual 3D Photography_Datasets/boatshed/stitch/front/disparities.png"
    depth_visual_file_path = depth_file_path + "_visual.png"

    # reader = png.Reader(depth_file_path)
    # pngdata = reader.read()

    # data = np.vstack(map(np.uint16, pngdata[2]))
    # data = np.array( map( np.uint16, pngdata[2] ))

    data = image_io.image_read(depth_file_path)
    empty_idx = np.all(data == 22, axis=-1)
    print(empty_idx)

    image_io.image_show(data.astype('float'))

    depth_visual = depth_io.depth_visual(data)
    depth_visual[empty_idx] = [255, 255, 255]

    image_io.image_save(depth_visual, depth_visual_file_path)


def file_vis(root_dir):
    """
    """
    of_data_path = pathlib.Path(root_dir)
    for file_path in of_data_path.iterdir():
        print(str(file_path))
        if file_path.suffix == ".floss":
            of_data = flow_io.readFlowFloss(str(file_path))
            of_data_color = flow_vis.flow_to_color(of_data)
            flow_visual_file_path = str(file_path) + "_vis.jpg"
            image_io.image_save(of_data_color, flow_visual_file_path)

        elif file_path.suffix == ".flo":
            of_data = flow_io.readFlowFile(str(file_path))
            of_data_color = flow_vis.flow_to_color(of_data)
            flow_visual_file_path = str(file_path) + "_vis.jpg"
            image_io.image_save(of_data_color, flow_visual_file_path)

        elif file_path.suffix == ".dpt":
            depth_visual_file_path = str(file_path) + "_vis.jpg"
            # data = image_io.image_read(str(file_path))
            data = depth_io.read_dpt(str(file_path))
            depth_visual = depth_io.depth_visual(data)
            image_io.image_save(depth_visual, depth_visual_file_path)


def test_warp():
    """
    """
    data_root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/replica_seq_data/"
    src_image_filepath = data_root_dir + "0001_pos2_rgb.jpg"
    # tar_image_filepath = data_root_dir + "0001_pos0_rgb.jpg"
    src_flo_forward_filepath = data_root_dir + "0001_pos2_opticalflow_forward.flo"
    src_image_warp_filepath = data_root_dir + "0001_pos2_rgb_forward_warp.jpg"

    src_image = image_io.image_read(src_image_filepath)
    # tar_image = image_io.image_read(tar_image_filepath)
    src_flo_forward = flow_io.readFlowFile(src_flo_forward_filepath)

    src_image_warp = flow_warp.warp_forward(src_image, src_flo_forward)
    image_io.image_save(src_image_warp, src_image_warp_filepath)


import numpy as np



def test_inside_polygon_2d():
    """[summary]
    """    
    # test 1: the points on the boundary inside of polygon 2d
    point_list = np.zeros((8, 2), dtype=np.float64)
    point_list[0, :] = [0, 0]
    point_list[1, :] = [0, 1.3]
    point_list[2, :] = [0.5, -1]
    point_list[3, :] = [-1, 0]
    point_list[4, :] = [1, 0]
    point_list[5, :] = [-1, -1.1]
    point_list[6, :] = [-1, 0.5]
    point_list[7, :] = [1, 0.5]

    polygon_points_list = np.zeros((4, 2), dtype=np.float64)
    polygon_points_list[0, :] = [-1, 1]
    polygon_points_list[1, :] = [1, 1]
    polygon_points_list[2, :] = [1, -1]
    polygon_points_list[3, :] = [-1, -1]

    from functools import reduce
    result = gp.inside_polygon_2d(point_list, polygon_points_list, True, eps=1e-6)
    print("Inside include on segment", result)
    result_gt = [True, False,  True,  True,  True, False,  True,  True]
    assert reduce((lambda x, y: x and y), result == result_gt)

    result = gp.inside_polygon_2d(point_list, polygon_points_list, False, eps=1e-15)
    print("Inside do not include on segment", result)
    result_gt = [True, False,  False,  False,  False, False,  False,  False]
    assert reduce((lambda x, y: x and y), result == result_gt)

    # test 2
    point_list_x, point_list_y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    point_list = np.stack((point_list_x.flatten(), point_list_y.flatten()), axis=1)
    polygon_points_list = np.zeros((4, 2), dtype=np.float64)
    polygon_points_list[0, :] = [-1, 1]
    polygon_points_list[1, :] = [1, 1]
    polygon_points_list[2, :] = [1, -1]
    polygon_points_list[3, :] = [-1, -1]
    result = gp.inside_polygon_2d(point_list, polygon_points_list, True).reshape(np.shape(point_list_x))
    image_io.image_show(result, verbose=True)

    # test 3
    point_list_x, point_list_y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
    point_list = np.stack((point_list_x.flatten(), point_list_y.flatten()), axis=1)
    polygon_points_list = np.zeros((3, 2), dtype=np.float64)
    polygon_points_list[0, :] = [0.,  0.76393202]
    polygon_points_list[1, :] = [0.66158454, -0.38196601]
    polygon_points_list[2, :] = [-0.66158454, -0.38196601]
    result = gp.inside_polygon_2d(point_list, polygon_points_list, True, 2/400).reshape(np.shape(point_list_x))
    image_io.image_show(result, verbose=False)

    # # test 4
    # gnomonic_proj_range = [-1, +1,-1, +1]
    # face_erp_x_grid = np.linspace(gnomonic_proj_range[0], gnomonic_proj_range[1], 50)
    # face_erp_y_grid = np.linspace(gnomonic_proj_range[2], gnomonic_proj_range[3], 50)
    #     face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
    # pbc = 1.0
    # gnomonic_bounding_box =  np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]])
    # available_list = gp.inside_polygon_2d(np.stack((face_erp_y_grid.flatten(), face_erp_x_grid.flatten()), axis=1), gnomonic_bounding_box).reshape(np.shape((50,50)))
    # image_io.image_show(available_list, verbose=False)

if __name__ == "__main__":
    # depth_visual()
    #root_folder = "D:/workdata/casual_stereo_vr_2020_test/boatshed_colmap_00_below_omni/Cache/29-2k-2k-DIS/"
    #root_folder = "D:/workdata/casual_stereo_vr_2020_test/bm_colmap_result_00_upper_omni/Cache/25-2k-2k-DIS/"
    # root_folder = "D:/workdata/casual_stereo_vr_2020_test/gasworks_colmap_00_below_omni/Cache/29-2k-2k-DIS/"
    # root_folder = "D:/workdata/casual_stereo_vr_2020_test/boatshed_colmap_00_below_omni_pano/Cache/29-2k-2k-DIS/"
    # of_vis(root_folder)
    # depth_map_format_transfrom()
    # root_folder = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/replica_seq_data/"
    # root_folder = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/replica_seq_data/"
    # file_vis(root_folder)

    # of_vis("/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/replica_seq_data/")
    # test_warp()
    test_inside_polygon_2d()