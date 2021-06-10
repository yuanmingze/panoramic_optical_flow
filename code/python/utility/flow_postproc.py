import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def erp_of_wraparound(erp_flow, of_u_threshold=None, of_v_threshold=None):
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
    index_u = flow_u > of_u_threshold
    flow_u[index_u] = flow_u[index_u] - image_width
    index_u = flow_u < -of_u_threshold
    flow_u[index_u] = flow_u[index_u] + image_width

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
    optical_flow_new = np.zeros_list(optical_flow, dtype=optical_flow.dtype)
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
