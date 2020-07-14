import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import flow_vis

def image_show(image):
    """
    visualize the numpy array
    """
    images = []
    cmap = plt.get_cmap('rainbow')
    fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
    axs.set_title('--')
    images.append(axs.imshow(image, cmap=cmap))
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
    plt.show()


def warp_forward(image_first, of_forward, padding_x=0, padding_y=0):
    """
    forward warpping
    The padding to protect the pixel warped range out of image boundary
    """
    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    dest_image = 255 * np.ones(
        (image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype=image_first.dtype)
    dest_image[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 +
                            padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 +
                            padding_y * 2, image_height + padding_y * 2)

    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = np.pad(of_forward[:, :, 0], ((
        padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((
        padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx = (x_idx + of_forward_x + 0.5).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx + of_forward_y + 0.5).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # 1) get new warpped image
    image_first_padded = np.pad(image_first, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_first_padded = image_first_padded[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    dest_image[y_idx, x_idx] = image_first_padded
    return dest_image


def warp_backward(image_target, of_forward, padding_x=0, padding_y=0):
    '''
    warp the target image to the source image with the forward optical flow.
    The padding is used in the case the optical flow warp out of the image range.
    '''
    image_size = np.shape(image_target)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    image_src = 255 * np.ones((image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype = image_target.dtype)
    image_src[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 + padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 + padding_y * 2, image_height + padding_y * 2)

    of_forward_x = np.pad(of_forward[:, :, 0], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    x_idx = (x_idx_tar + of_forward_x + 0.5).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    x_idx_tar = x_idx_tar.astype(np.int)
    x_idx_tar = x_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx_tar + of_forward_y + 0.5).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    y_idx_tar = y_idx_tar.astype(np.int)
    y_idx_tar = y_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # 1) get new warpped image
    image_target_padded = np.pad(image_target, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_src[y_idx_tar, x_idx_tar] = image_target_padded[y_idx, x_idx]
    return image_src


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
    # imgplot = plt.imshow(data[:,:,0])
    # plt.show()
    return data


def img_diff():
    image1 = np.array(Image.open("hotel_0_0000_pos0.jpeg"))
    image2 = np.array(Image.open("hotel_0_0000_pos1.jpeg"))
    diff = image1 - image2
    plt.imshow(diff)
    plt.show()


if __name__ == "__main__":
    root_dir = "/mnt/sda1/workdata/lightfield/GT-Replica-debug/hotel_0/replica_seq_data/"
    src_image_file = root_dir + "0001_rgb.jpg"
    tar_image_file = root_dir + "0002_rgb.jpg"
    of_file_path = root_dir + "/0001_opticalflow_forward.bin"

    # 1) load src image
    print("load source image from {}".format(src_image_file))
    src_image = np.array(Image.open(src_image_file))
    tar_image = np.array(Image.open(tar_image_file))

    height = np.shape(src_image)[0]
    width = np.shape(src_image)[1]
    print("image size is  {} x {}".format(height, width))

    # 2) load optical flow
    print("load optical flow from {}".format(of_file_path))
    of_forward = load_of_bin(of_file_path, width, height, False)
    # visualization optical flow
    of_forward_vis = flow_vis.flow_to_color(of_forward)
    image_show(of_forward_vis)

    # of_forward[:,:,1] = np.zeros(np.shape(of_forward)[0:2], dtype= np.float32)
    # #of_forward[:, int(width/2):,0] = 0 #np.zeros(np.shape(of_forward)[0:2], dtype= np.float32)
    # of_forward[:, :int(width/2),0] = 0

    # 3) warp source image with optical flow
    image_warped = warp_backward(tar_image, of_forward, 0, 0)
    # image_warped = warp_forward(src_image, of_forward, 100, 100)
    image_show(image_warped)
