from skimage.metrics import structural_similarity
import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

def sphercial_error_weight():
    # a cos weight
    # TODO
    pass


def ssim(image_0, image_1):
    # image_0_float = image_0.astype('float')
    # image_1_float = image_1.astype('float')
    return structural_similarity(image_0, image_1, multichannel=True)


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def mse(image_0, image_1):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_0.astype("float") - image_1.astype("float")) ** 2)
    err /= float(image_0.shape[0] * image_0.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def get_min_max(data, min_ratio=0.0, max_ratio=1.0):
    """Get the max and min based on the ratio. 

    :param data: data array.
    :type data: numpy
    :param min_ratio: The ratio of minimum value, defaults to 0.0
    :type min_ratio: float, optional
    :param max_ratio: The ratio of maximum value, defaults to 1.0
    :type max_ratio: float, optional
    """
    vmin_ = 0
    vmax_ = 0
    if min_ratio != 0.0 or max_ratio != 1.0:
        flow_array = data.flatten()
        vmin_idx = int(flow_array.size * min_ratio)
        vmax_idx = int((flow_array.size - 1) * max_ratio)
        vmin_ = np.partition(flow_array, vmin_idx)[vmin_idx]
        vmax_ = np.partition(flow_array, vmax_idx)[vmax_idx]
        if min_ratio != 0 or max_ratio != 1.0:
            log.warn("clamp the optical flow value form [{},{}] to [{},{}]".format(np.amin(data), np.amax(data), vmin_, vmax_))
    else:
        vmin_ = np.amin(data)
        vmax_ = np.amax(data)

    return vmin_, vmax_
