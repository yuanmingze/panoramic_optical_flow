
from utility import image_io
from skimage.transform import resize

import numpy as np

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def image_file_resize(image_input_filepath, image_output_filepath, resize_ratio=1.0):
    """[summary]

    :param image1_filepath: The input image file path.
    :type image1_filepath: str
    :param image2_filepath: The output image file path.
    :type image2_filepath: str
    """
    image_data = image_io.image_read(image_input_filepath)

    image_height = int(image_data.shape[0] * resize_ratio)
    image_width = int(image_data.shape[1] * resize_ratio)

    image_data_resized = resize(image_data, (image_height, image_width), anti_aliasing=True, preserve_range=True)

    image_io.image_save(image_data_resized, image_output_filepath)


def image_resize(image_data, image_size=None, image_ratio=None):
    """ Resize image

    :param image_data: [description]
    :type image_data: [type]
    :param image_size: [description], defaults to None
    :type image_size: [type], optional
    :param image_ratio: [description], defaults to None
    :type image_ratio: [type], optional
    :return: [description]
    :rtype: [type]
    """
    if image_size is None:
        image_size = [size * image_ratio for size in image_data.shape[:2]]
    image_rescaled = resize(image_data, image_size, anti_aliasing=True, preserve_range=True)
    return image_rescaled
