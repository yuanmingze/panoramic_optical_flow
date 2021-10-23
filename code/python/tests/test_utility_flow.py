import configuration as config

import pathlib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from utility import flow_vis
from utility import flow_io
from utility import flow_evaluate
from utility import image_io


def test_warp_image():
    """
    the demo of image warpping
    """
    root_dir = "../../data/replica_360_hotel_0/"
    src_image_file = root_dir + "0001_rgb.jpg"
    tar_image_file = root_dir + "0002_rgb.jpg"
    tar_warped_image_file = root_dir + "../0002_rgb_warped.jpg"
    # of_file_path = root_dir + "0001_opticalflow_forward.bin"

    of_forward_file_path = root_dir + "0001_opticalflow_forward.flo"
    of_backward_file_path = root_dir + "0002_opticalflow_backward.flo"

    # 1) load src image
    print("load source image from {}".format(src_image_file))
    src_image = np.array(Image.open(src_image_file))
    tar_image = np.array(Image.open(tar_image_file))

    height = np.shape(src_image)[0]
    width = np.shape(src_image)[1]
    print("image size is  {} x {}".format(height, width))

    # 2) load optical flow
    # print("load optical flow from {}".format(of_file_path))
    # of_forward = load_of_bin(of_file_path, width, height, False)

    # visualization optical flow
    of_forward = flow_io.readFlowFile(of_forward_file_path)
    of_forward_vis = flow_vis.flow_to_color(of_forward)
    image_io.image_show(of_forward_vis, "of_forward_vis")

    of_backward = flow_io.readFlowFile(of_backward_file_path)
    of_backward_vis = flow_vis.flow_to_color(of_backward)
    image_io.image_show(of_backward_vis, "of_forward_vis")

    # 3) warp source image with optical flow
    image_warped = flow_evaluate.warp_backward(tar_image, of_forward, 0, 0)
    # image_warped = warp_forward(src_image, of_forward, 100, 100)
    image_io.image_show(image_warped)
    image_io.image_save(image_warped, tar_warped_image_file)


def test_metrics():
    """
    the DEMO of optical flow evaluation
    """
    root_dir = "../../data/replica_360_hotel_0/"
    of_forward_file_path = root_dir + "0001_opticalflow_forward.flo"
    of_backward_file_path = root_dir + "0002_opticalflow_backward.flo"
    of_forward = flowio.readFlowFile(of_forward_file_path)
    of_backward = flowio.readFlowFile(of_backward_file_path)

    print("RMSE: 1) {}".format(flow_evaluate.RMSE(of_forward, of_forward)))
    print("RMSE: 2) {}".format(flow_evaluate.RMSE(of_forward, of_backward)))

    print("EPE: 1) {}".format(flow_evaluate.EPE(of_forward, of_forward)))
    print("EPE: 2) {}".format(flow_evaluate.EPE(of_forward, of_backward)))

    print("AAE: 1) {}".format(flow_evaluate.AAE(of_forward, of_forward)))
    print("AAE: 2) {}".format(flow_evaluate.AAE(of_forward, of_backward)))


def test_of_process_panorama():
    """

    """
    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/hotel_0/replica_seq_data/"
    of_file_path = root_dir + "/0002_opticalflow_backward.flo"
    of_data = flowio.readFlowFile(of_file_path)
    of_data_new = np.zeros(np.shape(of_data))
    # of_pano2ph(of_data, of_data_new)
    of_ph2pano(of_data, of_data_new)
    of_data_vis = flow_vis.flow_to_color(of_data_new)
    image_io.image_show(of_data_vis)


def test_of_warping_image():
    """
    test optical flow use warping images
    """
    data_root_folder = "/mnt/sda1/workdata/opticalflow_data/replica_360/hotel_0/replica_seq_data/"
    data_root_path = pathlib.Path(str(data_root_folder))
    #for item_file in data_root_path.iterdir():
    for file_idx in range(0, 17):
        # if not item_file.suffix == ".flo":
        #     continue

        # file_idx = int(str(item_file.name)[0:4])

        if file_idx % 5 == 0:
            print("test flow {}".format(file_idx))

        src_image_file_path = data_root_path / "{:04d}_rgb.jpg".format(file_idx)
        src_image = np.array(Image.open(src_image_file_path))
        src_image_warped_file_path = str(data_root_path / "{:04d}_rgb_forward_warped.jpg".format(file_idx))
        of_forward_file_path = data_root_path / "{:04d}_opticalflow_forward.flo".format(file_idx)

        tar_image_file_path = data_root_path / "{:04d}_rgb.jpg".format(file_idx + 1)
        tar_image = np.array(Image.open(tar_image_file_path))
        tar_image_warped_file_path = str(data_root_path / "{:04d}_rgb_backward_warped.jpg".format(file_idx + 1))
        of_backward_file_path = data_root_path / "{:04d}_opticalflow_backward.flo".format(file_idx + 1)

        # test forward optical flow & output
        of_forward = flow_io.readFlowFile(of_forward_file_path)
        src_image_warped = warp_forward(src_image, of_forward, 100, 100)
        image_io.image_save(src_image_warped, src_image_warped_file_path)

        # test backward optical flow & output
        of_backward = flow_io.readFlowFile(of_backward_file_path)
        tar_image_warped = warp_forward(tar_image, of_backward, 100, 100)
        image_io.image_save(tar_image_warped, tar_image_warped_file_path)


if __name__ == "__main__":

    test_warp_image()
    test_metrics()

    test_of_warping_image()
    # test_of_process_panorama()
    exit(0)

    #root_dir = "/mnt/sda1/workdata/lightfield/GT-Replica-debug/hotel_0/replica_seq_data/"
    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/hotel_0/replica_seq_data/"
    src_image_file = root_dir + "0002_rgb.jpg"
    tar_image_file = root_dir + "0001_rgb.jpg"
    of_file_path = root_dir + "/0002_opticalflow_backward.flo"
    image_warped_save_path = tar_image_file + "warped.jpg"

    # 1) load src image
    print("load source image from {}".format(src_image_file))
    src_image = np.array(Image.open(src_image_file))
    tar_image = np.array(Image.open(tar_image_file))

    height = np.shape(src_image)[0]
    width = np.shape(src_image)[1]
    print("image size is  {} x {}".format(height, width))

    # 2) load optical flow
    print("load optical flow from {}".format(of_file_path))
    # of_data, _ = load_of_bin(of_file_path, width, height, False)
    of_data = flowio.readFlowFile(of_file_path)
    # visualization optical flow
    of_data_vis = flow_vis.flow_to_color(of_data)
    image_show(of_data_vis)

    # of_data[:,:,1] = np.zeros(np.shape(of_data)[0:2], dtype= np.float32)
    # #of_data[:, int(width/2):,0] = 0 #np.zeros(np.shape(of_data)[0:2], dtype= np.float32)
    # of_data[:, :int(width/2),0] = 0

    # 3) warp source image with optical flow
    image_warped = warp_backward(tar_image, of_data, 0, 0)
    #image_warped = warp_forward(src_image, of_data, 0, 0)
    image_show(image_warped)
    image_save(image_warped, image_warped_save_path)
    image_diff(image_warped, src_image)
