import os
from struct import pack, unpack

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def image_read(image_file_path):
    """
    :return: return the numpy array of image
    """
    return np.asarray(Image.open(image_file_path))


def image_diff(image_generated, image_gt, output_path=""):
    """
    """
    # rgb to gray
    rgb_weights = [0.2989, 0.5870, 0.1140]

    image_generated_gray = np.dot(image_generated[..., :3], rgb_weights)
    image_gt_gray = np.dot(image_gt[..., :3], rgb_weights)

    # diff map to heatmap image
    diff = np.absolute(image_generated_gray - image_gt_gray)
    #plt.imshow(diff)
    #plt.show()
    image_show(diff)
    if output_path != "":
        plt.savefig(output_path)


def visual_data(data_array, verbose=False):
    """
    visualize the boolean or float etc. data to heatmap.
    """
    max = 1.0
    min = 0.0
    visualized_data = None
    if len(np.shape(data_array)) == 1 and data_array.dtype != np.bool:
        # is bool or float
        visualized_data = data_array
        max = np.max(data_array)
        min = np.min(data_array)
        if verbose:
            print("error_visual(): max error {}, min error {}".format(max, min))
    if len(np.shape(data_array)) == 1 and data_array.dtype == np.bool:
        visualized_data = data_array.astype(float)
        if verbose:
            print("error_visual(): max error {}, min error {}".format(max, min))

    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cmap = plt.get_cmap('jet')
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return (m.to_rgba(visualized_data)[:, :, :3] * 255).astype(np.uint8)


def image_show(image, verbose=True):
    """
    visualize the numpy array
    """
    if len(np.shape(image)) == 3:
        print("show 3 channels rgb image")
        image_rgb = image.astype(int)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()
    elif len(np.shape(image)) == 2:
        print("visualize 2 channel raw data")
        images = []
        cmap = plt.get_cmap('rainbow')
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
        axs.set_title('--')
        images.append(axs.imshow(image, cmap=cmap))
        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()
    elif len(np.shape(image)) == 1:
        print("show 1 channels data array")
        image_rgb = visual_data(image, verbose=False)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()

    else:
        print("the data channel is {}, should be visualized in advance.".format(len(np.shape(image))))


def image_save_rgba(image, image_file_path):
    """save the numpy array image to RGBA image.

    :param image: The RGBA image data, it request 4 channels.
    :type image: numpy
    :param image_file_path: The output image file path.
    :type image_file_path: str
    """
    # 0) check the file's extension
    _, file_extension = os.path.splitext(image_file_path)
    if file_extension.lower() != ".png":
        log.error("You are saving RGBA image to {}. The RGBA image store in *.png file.".format(image_file_path))
        return

    # 1) save to png file
    img = Image.fromarray(image)
    img = img.convert("RGBA")
    datas = img.getdata()

    ## set the the white pixel to transparent
    # newData = []
    # for item in datas:
    #     if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #         newData.append((255, 255, 255, 0))
    #     else:
    #         newData.append(item)

    img.putdata(datas)
    img.save(image_file_path, "PNG")


def image_save(image_data, image_file_path):
    """Save numpy array as image.

    :param image_data: Numpy array store image data. numpy 
    :type image_data: numpy
    :param image_file_path: The image's path
    :type image_file_path: str
    """
    # 0) convert the datatype
    image = None
    if image_data.dtype in [np.float, np.int64, np.int]:
        print("saved image array type is {}, converting to uint8".format(image_data.dtype))
        image = image_data.astype(np.uint8)
    else:
        image = image_data

    # 1) save to image file
    image_channels_number = image.shape[2]
    if image_channels_number == 4:
        image_save_rgba(image, image_file_path)
    elif image_channels_number == 3:
        im = Image.fromarray(image)
        im.save(image_file_path)
    else:
        log.error("The image channel number is {}".format(image_channels_number))
