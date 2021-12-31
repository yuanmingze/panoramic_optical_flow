
from skimage.transform import resize
import numpy as np

from . import image_io

from .logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def image_file_resize(image_input_filepath, image_output_filepath, image_height=None, image_width=None, resize_ratio=1.0):
    """ Resize image files.

    :param image_input_filepath: The input image file path.
    :type image_input_filepath: str
    :param image_output_filepath: The output image file path.
    :type image_output_filepath: str
    """
    image_data = image_io.image_read(image_input_filepath)

    if image_height is None or image_width is None:
        image_height = int(image_data.shape[0] * resize_ratio)
        image_width = int(image_data.shape[1] * resize_ratio)

    image_data_resized = resize(image_data, (image_height, image_width), anti_aliasing=True, preserve_range=True)

    image_io.image_save(image_data_resized, image_output_filepath)


def image_resize(image_data, image_size=None, image_ratio=None):
    """Resize image.

    :param image_data: The input image.
    :type image_data: numpy
    :param image_size: Resize image's size, defaults to None
    :type image_size: tuple, optional
    :param image_ratio: The resize ratio, defaults to None
    :type image_ratio: float, optional
    :return: The resized image.
    :rtype: numpy
    """
    if image_size is None:
        image_size = [size * image_ratio for size in image_data.shape[:2]]
    image_rescaled = resize(image_data, image_size, anti_aliasing=True, preserve_range=True)
    return image_rescaled


def get_erp_image_meshgrid(erp_height):
    """Get the erp image's mesh grid.

    :return: The erp image's meshgrid, [2, height, width]
    :rtype: numpy
    """
    erp_width = 2 * erp_height
    x_index = np.linspace(0, erp_width - 1, erp_width)
    y_index = np.linspace(0, erp_height - 1, erp_height)
    x_array, y_array = np.meshgrid(x_index, y_index)

    return np.stack((x_array, y_array))
