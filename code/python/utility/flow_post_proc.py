import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def erp_of_wraparound(erp_flow, of_u_threshold=None, of_v_threshold=None):
    """Convert the ERP optical flow without warp around to ERP optical flow.

    Replace the optical flow larger than threshold with wrap around value.

    :param erp_flow: the Non-ERP optical flow.
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
    index_u = flow_u < - of_u_threshold
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
    """ Convert the warp-around ERP optical flow to un-warp-around optical flow.

    process the panorama optical flow, change the warp around to NonERP optical flow.

    :param optical_flow: the panoramic optical flow
    :type optical_flow: numpy
    :return: the optical flow processed warp around
    :rtype: numpy
    """
    # TODO ignore the y warp around
    of_image_size = np.shape(optical_flow)
    image_height = of_image_size[0]
    image_width = of_image_size[1]

    if of_image_size[2] != 2:
        log.critical("The optical flow's channel number is {}.".format(of_image_size[2]))

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = optical_flow[:, :, 0]
    of_forward_y = optical_flow[:, :, 1]
    x_idx = (x_idx_tar + of_forward_x + 0.5).astype(np.int)
    y_idx = (y_idx_tar + of_forward_y + 0.5).astype(np.int)

    # 1) process the warp around
    optical_flow_new = np.zeros(optical_flow.shape, np.float)
    optical_flow_new[:] = optical_flow

    # process optical flow x
    x_idx_outrange_idx = np.where(x_idx >= image_width)
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx[x_idx_outrange_idx] - image_width
    x_idx_outrange_idx = np.where(x_idx < 0)
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx[x_idx_outrange_idx] + image_width

    # process optical flow y
    y_idx_outrange_idx = np.where(y_idx >= image_height)
    optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx[y_idx_outrange_idx] - image_height
    y_idx_outrange_idx = np.where(y_idx < 0)
    optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx[y_idx_outrange_idx] + image_height

    return optical_flow_new
