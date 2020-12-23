import numpy as np

import image_io
import flow_vis
import flow_io


def compute_occlusion():
    """
    
    """
    pass


def process_warp_around_spherical():
    """
    process the ward around of spherical coordinate system.
    The origin is in the center of image, the theta +0.5pi is on the top.

    :pararm phi : range is [-0.5 * pi, + 0.5 * pi)
    :param theta: range is [-pi, +pi)
    :return: corrected phi and theta
    """
    pass


def process_warp_around_erp(x, y, image_height):
    """
    The origian of ERP is on the TOP-Left of ERP image.

    :param x: the array of x 
    :param y: the array of y 
    :param image_height:
    :return : corrected x and y 
    """
    pass


def convert_warp_around(flow_original):
    """
    Process the optical flow warp around.

    :param flow: the flow without warp around
    :return: corrected flow
    """
    image_height = np.shape(flow_original)[0]
    image_width = np.shape(flow_original)[1]

    flow_u = flow_original[:,:,0]
    index_u = flow_u > (image_width / 2.0)
    flow_u[index_u] = flow_u[index_u] - image_width
    index_u = flow_u < -(image_width / 2.0)
    flow_u[index_u] = flow_u[index_u] + image_width

    flow_v = flow_original[:,:,1]
    index_v = flow_v > (image_height / 2.0)
    flow_v[index_v] = flow_v[index_v] - image_height
    index_v = flow_v < -(image_height / 2.0)
    flow_v[index_v] = flow_v[index_v] + image_height

    return np.stack((flow_u, flow_v), axis =2)




def of_ph2pano(optical_flow, optical_flow_new, of_warp_around_threshold=0.5):
    """
    process the warp around of optical flow.
    convert the pinhole type optical flow to panoramic optical flow.
    
    basic suppose: if optical flow in 
    :param:
    :param:
    """
    of_image_size = np.shape(optical_flow)
    image_height = of_image_size[0]
    image_width = of_image_size[1]
    image_channels = of_image_size[2]
    of_warp_around_threshold = image_width * of_warp_around_threshold

    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = optical_flow[:, :, 0]
    of_forward_y = optical_flow[:, :, 1]

    optical_flow_new[:] = optical_flow

    warp_around_idx = np.where(of_forward_x > of_warp_around_threshold)
    optical_flow_new[:, :, 0][warp_around_idx] = of_forward_x[warp_around_idx] - image_width

    warp_around_idx = np.where(of_forward_x < -of_warp_around_threshold)
    optical_flow_new[:, :, 0][warp_around_idx] = of_forward_x[warp_around_idx] + image_width


def of_pano2ph(optical_flow, optical_flow_new):
    """
    panoramic optical flow to pinhole optical flow.
    process the panorama optical flow, change the warp around to normal optical flow.

    :param: the panoramic optical flow
    :param: the optical flow processed warp around
    """
    of_image_size = np.shape(optical_flow)
    image_height = of_image_size[0]
    image_width = of_image_size[1]
    image_channels = of_image_size[2]

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = optical_flow[:, :, 0]
    of_forward_y = optical_flow[:, :, 1]
    x_idx = (x_idx_tar + of_forward_x + 0.5).astype(np.int)
    y_idx = (y_idx_tar + of_forward_y + 0.5).astype(np.int)

    # 1) process the warp around
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

