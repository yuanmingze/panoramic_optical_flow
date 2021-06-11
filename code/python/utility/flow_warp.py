import numpy as np
from scipy import ndimage

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def warp_backward(image_target, of_forward):
    """
    Backward warp with optical flow from the target image to generate the source image. 

    :param image_target: The terget image of optical flow, [height, width, channel].
    :type image_target: numpy
    :param of_forward:  optical flow from source to target, [height, width, 2].
    :type of_forward: numpy
    :return: Generated source image.
    :rtype: numpy
    """
    image_height = image_target.shape[0]
    image_width = image_target.shape[1]
    image_channels = None
    if len(image_target.shape) == 3:
        image_channels = image_target.shape[2]
    elif len(image_target.shape) == 2:
        image_channels = None
    else:
        log.error("The image shape is {}, do not support.".format(image_target.shape))
    dest_image = np.zeros_like(image_target, dtype=image_target.dtype)

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    # x_idx_new = np.remainder(x_idx + of_forward[:, :, 0], image_width)
    # y_idx_new = np.remainder(y_idx + of_forward[:, :, 1], image_height)
    x_idx_new = (x_idx + of_forward[:, :, 0])
    y_idx_new = (y_idx + of_forward[:, :, 1])

    if image_channels is not None:
        for channel_index in range(0, image_channels):
            dest_image[y_idx.astype(int), x_idx.astype(int), channel_index] = ndimage.map_coordinates(image_target[:, :, channel_index], [y_idx_new, x_idx_new], order=1, mode='wrap')
    else:
        dest_image[y_idx.astype(int), x_idx.astype(int)] = ndimage.map_coordinates(image_target[:, :], [y_idx_new, x_idx_new], order=1, mode='constant', cval=255)

    return dest_image


def warp_forward_padding(image_target, of_forward, padding_x=0, padding_y=0):
    '''
    warp the target image to the source image with the forward optical flow.
    The padding is used in the case the optical flow warp out of the image range.
    '''
    image_size = np.shape(image_target)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    image_src = 255 * np.ones((image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype=image_target.dtype)
    image_src[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 + padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 + padding_y * 2, image_height + padding_y * 2)

    of_forward_x = np.pad(of_forward[:, :, 0], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    x_idx = (x_idx_tar + of_forward_x).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    x_idx_tar = x_idx_tar.astype(np.int)
    x_idx_tar = x_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx_tar + of_forward_y).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    y_idx_tar = y_idx_tar.astype(np.int)
    y_idx_tar = y_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # check the range of x_idx & y_idx
    if not np.logical_and(x_idx_tar >= 0, x_idx_tar < image_width).all():
        print("image warp x_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return
    if not np.logical_and(y_idx_tar >= 0, y_idx_tar < image_height).all():
        print("image warp y_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return

    # 1) get new warpped image
    image_target_padded = np.pad(image_target, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_src[y_idx_tar, x_idx_tar] = image_target_padded[y_idx, x_idx]
    return image_src


def warp_forward(image_first, of_forward, wrap_around=False, ignore_transparent=False):
    """ forward warp image with optical flow. 

    :param image_first: input image, when it's 4 channels image, use the alpha channel to ignore the transparent area [height,width,:].
    :type image_first: numpy
    :param of_forward: forward optical flow. [height, width,  2]
    :type of_forward: numpy
    :param wrap_around: whether process the wrap around pixels, defaults to False
    :type wrap_around: bool, optional
    :param ignore_transparent: if yes do not warp the transparent are in the first image, defaults to False
    :type ignore_transparent: bool, optional
    :return: warped image
    :rtype: numpy
    """
    valid_pixels_index = None
    if image_first.shape[2] == 4:
        # RGBA images, ignore the transparent area
        valid_pixels_index = image_first[:, :, 3] == 255

    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    # x_idx, y_idx = np.mgrid[0:image_height, 0: image_width]
    x_idx_new = x_idx + of_forward[:, :, 0]
    y_idx_new = y_idx + of_forward[:, :, 1]

    # check index out of the image bounds
    if wrap_around:
        x_idx_new = np.where(x_idx_new > 0, x_idx_new, x_idx_new + image_width)
        x_idx_new = np.where(x_idx_new < image_width, x_idx_new, np.remainder(x_idx_new, image_width))
        y_idx_new = np.where(y_idx_new > 0, y_idx_new, y_idx_new + image_height)
        y_idx_new = np.where(y_idx_new < image_height, y_idx_new, np.remainder(y_idx_new, image_height))
    else:
        x_idx_new = np.where(x_idx_new > 0, x_idx_new, 0)
        x_idx_new = np.where(x_idx_new < image_width, x_idx_new, image_width - 1)
        y_idx_new = np.where(y_idx_new > 0, y_idx_new, 0)
        y_idx_new = np.where(y_idx_new < image_height, y_idx_new, image_height - 1)

    if valid_pixels_index is not None:
        x_idx_new = x_idx_new[valid_pixels_index]
        y_idx_new = y_idx_new[valid_pixels_index]
        x_idx = x_idx[valid_pixels_index]
        y_idx = y_idx[valid_pixels_index]

    x_idx_new = x_idx_new.astype(int)
    y_idx_new = y_idx_new.astype(int)
    dest_image = np.zeros(np.shape(image_first), dtype=image_first.dtype)
    for channel_index in range(0, image_channels):
        dest_image[y_idx_new, x_idx_new, channel_index] = ndimage.map_coordinates(image_first[:, :, channel_index], [y_idx, x_idx], order=1, mode='constant', cval=255)

    return dest_image


def warp_forward_padding(image_first, of_forward, padding_x=0, padding_y=0):
    """
    forward warpping
    The padding to protect the pixel warped range out of image boundary
    """
    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    dest_image = 255 * np.ones((image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype=image_first.dtype)
    dest_image[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 + padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 + padding_y * 2, image_height + padding_y * 2)

    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = np.pad(of_forward[:, :, 0], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx = (x_idx + of_forward_x).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx + of_forward_y).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # check the range of x_idx & y_idx
    if not np.logical_and(x_idx >= 0, x_idx < image_width).all():
        print("image warp x_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return
    if not np.logical_and(y_idx >= 0, y_idx < image_height).all():
        print("image warp y_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return

    # 1) get new warpped image
    image_first_padded = np.pad(image_first, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_first_padded = image_first_padded[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    dest_image[y_idx, x_idx] = image_first_padded
    return dest_image
