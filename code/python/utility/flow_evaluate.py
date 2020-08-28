
import math
import numpy as np
from scipy import ndimage


"""
functions used to evaluate the quality of optical flow;
"""

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def EPE(of_ground_truth, of_evaluation):
    """
    endpoint error (EE)
    reference : https://github.com/prgumd/GapFlyt/blob/master/Code/flownet2-tf-umd/src/flowlib.py

    return: average of EPE & average 
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # remove the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    index_su = su[ind2]
    index_sv = sv[ind2]
    # an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    # un = index_su * an
    # vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    # tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    # tun = index_stu * tn
    # tvn = index_stv * tn

    # angle = un * tun + vn * tvn + (an * tn)
    # index = [angle == 1.0]
    # angle[index] = 0.999
    # ang = np.arccos(angle)
    # mang = np.mean(ang)
    # mang = mang * 180 / np.pi

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)

    return mepe


def RMSE(of_ground_truth, of_evaluation):
    """
    compute the root mean square error(RMSE) of optical flow

    retrun: rmse
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # ignore the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0

    ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)

    diff_u = stu[ind2] - su[ind2]
    diff_v = stv[ind2] - sv[ind2]

    rmse = np.sqrt(np.sum(diff_u ** 2 + diff_v ** 2) / np.shape(diff_u)[0])
    return rmse


def AAE(of_ground_truth, of_evaluation):
    """
    The average angular error(AAE) 

    The result is between 0 and 2 * PI.
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # remove the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    index_su = su[ind2]
    index_sv = sv[ind2]

    index_stu = stu[ind2]
    index_stv = stv[ind2]

    # compute the average angle
    uv_cross = index_su * index_stv - index_stu * index_sv
    uv_dot = index_su * index_stu + index_stv * index_sv
    angles = np.arctan2(uv_cross, uv_dot)

    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)
    if x == 0 and y == 0:
        raise ValueError(
            "The angle average of the inputs is undefined: %r" % angles)

    return math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi) * 180 / np.pi


def warp_backward(image_target, of_forward):
    """
    forward warp with optical flow. 
    warp image with interpolation, scipy.ndimage.map_coordinates
    """
    image_size = np.shape(image_target)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    dest_image = np.zeros(np.shape(image_target), dtype=image_target.dtype)

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)

    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    x_idx_new = (x_idx + of_forward[:, :, 0])
    y_idx_new = (y_idx + of_forward[:, :, 1])

    for channel_index in range(0, image_channels):
        dest_image[y_idx.astype(int), x_idx.astype(int), channel_index] = ndimage.map_coordinates(image_target[:, :, channel_index], [y_idx_new, x_idx_new], order=1, mode='constant', cval=255)

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


def warp_forward(image_first, of_forward):
    """
    forward warp with optical flow. 
    warp image with interpolation, scipy.ndimage.map_coordinates
    """
    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    dest_image = np.zeros(np.shape(image_first), dtype=image_first.dtype)

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)

    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    x_idx_new = (x_idx + of_forward[:, :, 0]).astype(int)
    y_idx_new = (y_idx + of_forward[:, :, 1]).astype(int)

    # check index out of the image bounds
    x_idx_new = np.where(x_idx_new > 0, x_idx_new, 0)
    x_idx_new = np.where(x_idx_new < image_width - 1, x_idx_new, image_width - 1)

    y_idx_new = np.where(y_idx_new > 0, y_idx_new, 0)
    y_idx_new = np.where(y_idx_new < image_height - 1, y_idx_new, image_height - 1)

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
