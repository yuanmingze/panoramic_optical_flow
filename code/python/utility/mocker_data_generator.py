import configuration

import numpy as np
from scipy import ndimage


def hsv_to_rgb(hsv):
    """
    Code from https://gist.github.com/PolarNick239/691387158ff1c41ad73c.

    >>> from colorsys import hsv_to_rgb as hsv_to_rgb_single
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.79, 239))
    'r=50 g=126 b=239'
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.25, 0.35, 200.0))
    'r=165 g=200 b=130'
    >>> np.set_printoptions(0)
    >>> hsv_to_rgb(np.array([[[0.60, 0.79, 239], [0.25, 0.35, 200.0]]]))
    array([[[  50.,  126.,  239.],
            [ 165.,  200.,  130.]]])
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.0, 239))
    'r=239 g=239 b=239'
    >>> hsv_to_rgb(np.array([[0.60, 0.79, 239], [0.60, 0.0, 239]]))
    array([[  50.,  126.,  239.],
           [ 239.,  239.,  239.]])
    """
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    return rgb.reshape(input_shape)


def image_strip(image_height=300, image_width=600):
    """ Get a mocking data, size is [height, width, 3]
    
    :return: the mocking data.
    :rtype: numpy
    """
    data = np.arange(image_height * image_width) / (image_height * image_width)

    hsv_data = np.stack((data, np.ones_like(data), np.ones_like(data)), axis=1)
    rgb_data = hsv_to_rgb(hsv_data)
    rgb_data = (rgb_data * 255.0).astype(np.uint8)
    rgb_data = rgb_data.reshape((image_height, image_height * 2, 3))
    return rgb_data


def image_square(image_height=300, image_width=600):
    data = np.zeros((image_height, image_width), np.float64)
    data[5:-5, 5:-5] = 1.0
    data = ndimage.distance_transform_bf(data)
    max_value = np.max(data)
    min_value = np.min(data)
    data = (data - min_value) / (max_value - min_value)
    hsv_data = np.stack((data, np.ones_like(data), np.ones_like(data)), axis=2).reshape((-1, 3))
    rgb_data = hsv_to_rgb(hsv_data)
    rgb_data = (rgb_data * 255.0).astype(np.uint8)
    rgb_data = rgb_data.reshape((image_height, image_width, 3))
    return rgb_data


def opticalflow_simple(erp_image_height=20, erp_image_width=40, u_default=10, v_default=10):
    """ Create the mock optical flow data.

    :param erp_image_height: [2, height, width], float 64
    :type erp_image_height: [type]
    """
    of_data = np.zeros((2, erp_image_height, erp_image_width), np.float64)
    of_data[0, :, :] = u_default
    of_data[1, :, :] = v_default
    return of_data


def opticalflow_random(image_height=20, image_width=40, u_value=None, v_value=None):
    """ Return [height, width, 2]"""
    if u_value is not None and v_value is not None:
        of_data = np.zeros((image_height, image_width, 2), np.float64)

        of_data[0, :, :] = u_value
        of_data[1, :, :] = v_value
    else:
        rng = np.random.default_rng(12345)
        of_data = rng.standard_normal(size=(image_height, image_width, 2), dtype=np.float64)
        of_data[:, :, 0] = of_data[:, :, 0] * image_width * 0.5
        of_data[:, :, 1] = of_data[:, :, 1] * image_height * 0.5
    return of_data
