import os

import numpy as np

import png
import itertools

from PIL import Image

from utility import depth_io
from utility import image_io


path = os.getcwd()
TEST_DATA_ROOT_FOLDER = path + "/../../data/replica_360/office_0/"
TEMP_DATA_ROOT_FOLDER = path + "/../../data/test_temp/"


def test_detph_io():
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
    #import ipdb; ipdb.set_trace()


    image_io.image_show(data.astype('float'))

    depth_visual = depth_io.depth_visual(data)
    depth_visual[empty_idx] = [255, 255, 255]

    image_io.image_save(depth_visual, depth_visual_file_path)



if __name__ == "__main__":
    depth_visual()
