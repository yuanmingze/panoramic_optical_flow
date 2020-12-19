
import numpy as np

import cv2


from .logger import Logger

log = Logger(__name__)
log.logger.propagate = False

def DIS(image_src, image_tar):
    """Compute the DIS flow.

    :param image_src: The optical flow source image.
    :type image_src: numpy
    :param image_tar: [description]
    :type image_tar: [type]
    :return: [description]
    :rtype: [type]
    """
    if image_src.shape[2] == 3:
        image_src_gray  = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_src_gray = image_src

    if image_tar.shape[2] == 3:
        image_tar_gray = cv2.cvtColor(image_tar, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_tar_gray = image_tar

    inst = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(True)

    return inst.calc(image_src_gray, image_tar_gray, None)
