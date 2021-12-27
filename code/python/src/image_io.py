import os
from struct import pack, unpack

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import depth_io

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def image_read(image_file_path):
    """Load rgb image data from file.

    :param image_file_path: the absolute path of image
    :type image_file_path: str
    :return: the numpy array of image
    :rtype: numpy
    """
    if not os.path.exists(image_file_path):
        log.error("{} do not exist.".format(image_file_path))

    return np.asarray(Image.open(image_file_path))


def image_diff(image_generated, image_gt, output_path=""):
    """Generate image different color map.
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


def visual_data(data_array):
    """Visualize the single channel boolean or float etc. data to heatmap.
    """
    return depth_io.depth_visual_save(data_array)
    # max = None
    # min = None
    # visualized_data = None
    # if len(np.shape(data_array)) == 2 and data_array.dtype != np.bool:
    #     visualized_data = data_array
    #     min, max = image_evaluate.get_min_max(data_array, min_ratio, max_ratio)
    #     log.debug("error_visual(): max error {}, min error {}".format(max, min))
    # elif len(np.shape(data_array)) == 2 and data_array.dtype == np.bool:
    #     visualized_data = data_array.astype(float)
    #     max = 1.0
    #     min = 0.0
    #     log.debug("error_visual(): max error {}, min error {}".format(max, min))
    # else:
    #     log.error("Data shape is {}, just support single channel data.".format(data_array.shape))

    # fig, axes = plt.subplots()
    # axes.axis("off")
    # norm = mpl.colors.Normalize(vmin=min, vmax=max)
    # smap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # fig.colorbar(mappable=smap)
    # axes.imshow(smap.to_rgba(visualized_data))
    # # plt.show()
    # fig.canvas.draw()  # draw the renderer
    # w, h = fig.canvas.get_width_height()     # Get the RGBA buffer from the figure
    # buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    # buf.shape = (w, h, 4)
    # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)
    # image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # image = np.asarray(image)
    # return image[:, :, 0:3]


def image_show(image, title=" ", verbose=False):
    """Visualize the numpy array
    """
    if len(np.shape(image)) == 3:
        if verbose: print("show 3 channels rgb image")
        image_rgb = image.astype(int)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()
    elif len(np.shape(image)) == 2:
        if verbose: print("visualize 2 channel raw data")
        images = []
        cmap = plt.get_cmap('rainbow')
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
        axs.set_title(title)
        images.append(axs.imshow(image, cmap=cmap))
        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()
    elif len(np.shape(image)) == 1:
        print("show 1 channels data array")
        image_rgb = visual_data(image)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()

    else:
        print("the data channel is {}, should be visualized in advance.".format(len(np.shape(image))))


def image_save_rgba(image, image_file_path):
    """Save the numpy array image to RGBA image.

    :param image: The RGBA image data, it request 4 channels.
    :type image: numpy
    :param image_file_path: The output image file path.
    :type image_file_path: str
    """
    # 0) check the file's extension
    _, file_extension = os.path.splitext(image_file_path)
    if file_extension.lower() != ".png":
        log.error("Saving RGBA image to {}. The RGBA image store in *.png file.".format(image_file_path))
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


def image_seq2gif(image_seq_list, output_filepath):
    """ Convert a image sequence to a gif animation image and save. 

    :param image_seq_list: The input image sequence of numpy image data.
    :type image_seq_list: list 
    :param output_filepath: The output gif image's file path.
    :type output_filepath: str
    """
    if len(image_seq_list) <= 0:
        log.error("image seq is empty!")

    image_width = image_seq_list[0].shape[0]
    image_height = image_seq_list[0].shape[1]

    image_seq_list_ = None
    if type(image_seq_list[0]) == np.ndarray:
        image_seq_list_ = []
        for image in image_seq_list:
            image_seq_list_.append(Image.fromarray(image))
    else:
        image_seq_list_ = image_seq_list

    # image_gif = Image.new('RGB', (image_height, image_width), (0, 0, 0))
    image_gif = image_seq_list_[0]

    image_gif.save(output_filepath,
                   save_all=True,  # save all frame to gif image
                   append_images=image_seq_list_,  # the image to be saved
                   optimize=False,  # compress
                   duration=40,  # each frame display time, MS (milliseconds)
                   loop=0  # 0 is loop forever
                   )
