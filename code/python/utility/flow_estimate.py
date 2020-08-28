
import numpy as np

import cv2


def DIS(image_src, image_tar):
    """

    """
    image_src_gray  = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    image_tar_gray = cv2.cvtColor(image_tar, cv2.COLOR_BGR2GRAY)


    inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(True)

    flow = None
    return inst.calc(image_src_gray, image_tar_gray, None)
