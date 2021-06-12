import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def erp_pixles_modulo(x_arrray, y_array, image_width, image_height):
    """ Make x,y and ERP pixels coordinate system range.
    """
    x_arrray_new = np.remainder(x_arrray + 0.5, image_width) - 0.5
    y_array_new = np.remainder(y_array + 0.5, image_height) - 0.5
    return x_arrray_new, y_array_new


def erp_of_wraparound(erp_flow, of_u_threshold=None):
    """
    Convert un-wrap-around (do not overflow) to ERP optical flow to the wrap-around (overflow) ERP optical flow.
    The optical flow larger than threshold need to be wrap-around.

    :param erp_flow: the panoramic optical flow [height, width, 2]
    :type  erp_flow: numpy
    :param of_u_threshold: The wrap-around threshold of optical flow u.
    :type of_u_threshold: float
    :param of_v_threshold: The wrap-around threshold of optical flow v.
    :type of_v_threshold: float
    :return: The ERP optical flow
    :rtype: numpy
    """
    image_width = np.shape(erp_flow)[1]
    if of_u_threshold is None:
        of_u_threshold = image_width / 2.0

    flow_u = erp_flow[:, :, 0]
    # minus width
    index_minus_src_point_range = np.full(flow_u.shape[0:2], False)
    index_minus_src_point_range[:, 0: int(image_width - 1 - of_u_threshold)] = True
    index_minus_src_point = np.logical_and(index_minus_src_point_range, flow_u > of_u_threshold)
    flow_u[index_minus_src_point] = flow_u[index_minus_src_point] - image_width
    # plus width
    index_plus_src_point_range = np.full(flow_u.shape, False)
    index_plus_src_point_range[:, int(image_width - 1 - of_u_threshold): image_width - 1] = True
    index_plus_src_point = np.logical_and(index_plus_src_point_range, flow_u < -of_u_threshold)
    flow_u[index_plus_src_point] = flow_u[index_plus_src_point] + image_width

    # image_height = np.shape(erp_flow)[0]
    # if of_v_threshold is None:
    #     of_v_threshold = (image_height - 1.0) / 2.0
    # flow_v = erp_flow[:, :, 1]
    # index_v = flow_v > of_v_threshold
    # flow_v[index_v] = flow_v[index_v] - image_height
    # index_v = flow_v < -of_v_threshold
    # flow_v[index_v] = flow_v[index_v] + image_height
    return np.stack((flow_u, erp_flow[:, :, 1]), axis=2)


def erp_of_unwraparound(optical_flow):
    """
    Convert the wrap-around (overflow) ERP optical flow to un-warp-around (do not overflow) optical flow.

    :param optical_flow: the panoramic optical flow [height, width, 2]
    :type optical_flow: numpy
    :return: the optical flow processed warp around
    :rtype: numpy
    """
    if len(optical_flow.shape) != 3 or optical_flow.shape[2] != 2:
        log.error("The optical flow's channel number is {}.".format(optical_flow.shape))

    image_height = np.shape(optical_flow)[0]
    image_width = np.shape(optical_flow)[1]
    optical_flow_new = np.zeros_like(optical_flow, dtype=optical_flow.dtype)
    optical_flow_new[:] = optical_flow

    # 0) comput new location
    x_idx_src = np.linspace(0, image_width - 1, image_width)
    y_idx_src = np.linspace(0, image_height - 1, image_height)
    x_idx_src, y_idx_src = np.meshgrid(x_idx_src, y_idx_src)
    x_idx_tar = x_idx_src + optical_flow[:, :, 0]
    # y_idx_tar = y_idx_src + optical_flow[:, :, 1]

    # 1) process the optical flow x warp around
    x_idx_outrange_idx = np.where(x_idx_tar >= (image_width - 0.5))
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx_tar[x_idx_outrange_idx] - image_width
    x_idx_outrange_idx = np.where(x_idx_tar < -0.5)
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx_tar[x_idx_outrange_idx] + image_width

    # # process optical flow y
    # y_idx_outrange_idx = np.where(y_idx_tar >= image_height)
    # optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx_tar[y_idx_outrange_idx] - image_height
    # y_idx_outrange_idx = np.where(y_idx_tar < 0)
    # optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx_tar[y_idx_outrange_idx] + image_height

    return optical_flow_new
