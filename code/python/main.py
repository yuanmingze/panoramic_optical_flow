import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import flow_vis
import flowio
import flow_evaluate

def image_show(image, image_title= "Image"):
    """
    visualize the numpy array
    """
    images = []
    cmap = plt.get_cmap('rainbow')
    fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
    axs.set_title(image_title)
    images.append(axs.imshow(image, cmap=cmap))
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
    plt.show()


def image_save(image, image_file_path):
    """ 
    save the numpy as image
    """
    im = Image.fromarray(image)
    im.save(image_file_path)


def load_of_bin(binary_file_path, height, width, visual_enable=True):
    """
    load depth value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(width, height, 4)

    if visual_enable:
        cmap = plt.get_cmap('rainbow')
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

        images = []
        axs[0].set_title('optical flow x')
        images.append(axs[0].imshow(data[:, :, 0], cmap=cmap))

        axs[1].set_title('optical flow y')
        images.append(axs[1].imshow(data[:, :, 1], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    if visual_enable:
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))

        images = []
        axs.set_title('optical flow x')
        images.append(axs.imshow(data[:, :, 0], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    if visual_enable:
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))

        images = []
        axs.set_title('optical flow y')
        images.append(axs.imshow(data[:, :, 1], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    return data[:, :, 0:2]


def load_depth_bin(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(height, width, 1)
    return data


def img_diff(image_1st, image_2nd):
    diff = image_1st - image_2nd
    plt.imshow(diff)
    plt.show()


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
    of_forward = flowio.readFlowFile(of_forward_file_path)
    of_forward_vis = flow_vis.flow_to_color(of_forward)
    image_show(of_forward_vis, "of_forward_vis")

    of_backward = flowio.readFlowFile(of_backward_file_path)
    of_backward_vis = flow_vis.flow_to_color(of_backward)
    image_show(of_backward_vis, "of_forward_vis")

    # 3) warp source image with optical flow
    image_warped = flow_evaluate.warp_backward(tar_image, of_forward, 0, 0)
    # image_warped = warp_forward(src_image, of_forward, 100, 100)
    image_show(image_warped)
    image_save(image_warped, tar_warped_image_file)


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


if __name__ == "__main__":
    test_warp_image()
    test_metrics()

