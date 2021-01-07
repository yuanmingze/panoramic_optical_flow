import numpy as np
from scipy import ndimage

import gnomonic_projection
from scipy.stats import norm

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def get_blend_weight(face_x_src_gnomonic, face_y_src_gnomonic, weight_type, flow_uv=None, image_erp_src=None, image_erp_tar=None):
    """Compute the faces's weight.

    :param face_x_src_gnomonic: the pixel's gnomonic coordinate x in tangent image
    :type face_x_src_gnomonic: numpy, [pixel_number]
    :param face_y_src_gnomonic: the pixel's gnomonic coordinate y in tangent image
    :type face_y_src_gnomonic: numpy, [pixel_number]
    :param weight_type: The weight compute method, [straightforward|cartesian_distance]
    :type weight_type: str
    :param flow_uv: the tangent face optical flow which is in image coordinate, unit is pixel.
    :type flow_uv: numpy 
    :param image_erp_src: the source ERP rgb image, used to compute the optical flow warp error
    :type: numpy
    :param image_erp_tar: the target ERP rgb image used to compute the optical flow warp error
    :type: numpy
    :return: the cubemap's face weight used to blend different faces to ERP image.
    :rtype: numpy
    """
    weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)

    if weight_type == "straightforward":
        # just set the pixels in this cube map face range is available. [-1, +1, -1, +1]
        pbc = 1
        gnomonic_bounding_box = np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]])
        available_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box)
        weight_map[available_list] = 1.0
    elif weight_type == "cartesian_distance_log":
        radius = np.linalg.norm(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), axis=1)
        radius_log = np.log(radius + 1.1) / np.log(10)
        # zeros_index = (radius == 0)
        # radius[zeros_index] = np.finfo(np.float).eps
        weight_map = 1.0 / radius_log
    elif weight_type == "cartesian_distance_exp":
        radius = np.linalg.norm(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), axis=1)
        radius_log = np.exp(radius - 4)
        weight_map = 1.0 / radius_log
    elif weight_type == "normal_distribution":
        center_point_x = 0.0
        center_point_y = 0.0
        mean = 0.5
        stdev = 1
        radius = np.linalg.norm(np.stack((face_x_src_gnomonic - center_point_x, face_y_src_gnomonic - center_point_y), axis=1), axis=1)
        weight_map = (1.0 / (stdev * np.sqrt(2*np.pi))) * np.exp(-0.5*((radius - mean) / stdev) ** 2)
    elif weight_type == "normal_distribution_flowcenter":
        # TODO use the optical flow average to compute the center point
        center_point_x = flow_uv[:, 0].mean() / (0.5 * np.sqrt(face_x_src_gnomonic.shape[0]))  # form pixel to gnomonic
        center_point_y = flow_uv[:, 1].mean() / (0.5 * np.sqrt(face_x_src_gnomonic.shape[0]))
        mean = 0.0
        stdev = 1
        radius = np.linalg.norm(np.stack((face_x_src_gnomonic - center_point_x, face_y_src_gnomonic - center_point_y), axis=1), axis=1)
        weight_map = (1.0 / (stdev * np.sqrt(2*np.pi))) * np.exp(-0.5*((radius - mean) / stdev) ** 2)
    elif weight_type == "image_warp_error":
        # compute the weight base on the ERP RGB image warp match.
        # flow_uv: target image's pixels coordinate corresponding the warpped pixels
        pixels_number = face_x_src_gnomonic.shape[0]
        channel_number = image_erp_src.shape[2]
        image_erp_tar_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_src_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_warp_diff = np.zeros((pixels_number, channel_number), np.float)
        for channel in range(0, channel_number):
            image_erp_tar_flow[:, channel] = ndimage.map_coordinates(image_erp_tar[:, :, channel], [flow_uv[:, 1], flow_uv[:, 0]], order=1, mode='constant', cval=255)
            image_erp_src_flow[:, channel] = ndimage.map_coordinates(image_erp_src[:, :, channel], [face_y_src_gnomonic, face_x_src_gnomonic], order=1, mode='constant', cval=255)
            image_erp_warp_diff[:, channel] = np.absolute(image_erp_tar_flow[:, channel] - image_erp_src_flow[:, channel])

        rgb_diff = np.linalg.norm(image_erp_warp_diff, axis=1)
        non_zeros_index = rgb_diff != 0.0
        weight_map = np.ones(face_x_src_gnomonic.shape[0], dtype=np.float)
        weight_map[non_zeros_index] = 0.95 / rgb_diff[non_zeros_index]
    else:
        log.error("the weight method {} do not exist.".format(weight_type))
    return weight_map


def flow2offset(erp_flow, use_weight=True):
    """Compute the  two image rotation from the ERP image's optical flow.

    :param erp_flow: the erp image's flow 
    :type erp_flow: numpy 
    :return: the offset of ERP image, [longitude shift, latitude shift
    :rtype: float
    """
    erp_image_height = erp_flow.shape[0]
    erp_image_width = erp_flow.shape[1]

    # convert the pixel offset to rotation radian
    azimuth_delta_array = 2.0 * np.pi * (erp_flow[:, :, 0] / erp_image_width)
    elevation_delta_array = np.pi * (erp_flow[:, :, 1] / erp_image_height)

    if use_weight:
        # weight of the u, width
        stdev = erp_image_height * 0.5 * 0.25
        weight_u_array_index = np.arange(erp_image_height)
        weight_u_array = norm.pdf(weight_u_array_index, erp_image_height / 2.0, stdev)
        azimuth_delta_array = np.average(azimuth_delta_array, axis=0, weights=weight_u_array)

        # weight of the v, height
        stdev = erp_image_width * 0.5 * 0.25
        weight_v_array_index = np.arange(erp_image_width)
        weight_v_array = norm.pdf(weight_v_array_index, erp_image_width / 2.0, stdev)
        elevation_delta_array = np.average(elevation_delta_array, axis=1,  weights=weight_v_array)

    azimuth_delta = np.mean(azimuth_delta_array)
    elevation_delta = np.mean(elevation_delta_array)

    return azimuth_delta, elevation_delta


def image_align(erp_image_src, erp_flow):
    """Rotate the source image base on the flow.

    :param erp_image_src: the flow's source image
    :type erp_image_src: numpy 
    :param erp_flow: the erp image's flow, the optical flow is not processed the wrap around.
    :type erp_flow: numpy 
    :return: The rotated source image
    :rtype: numpy
    """
    # 0) compuate the average of optical flow & get the delta phi and lambda
    azimuth_delta, elevation_delta = flow2offset(erp_flow)

    from scipy.spatial.transform import Rotation as R
    rotation_matrix = R.from_euler("xyz", [np.degrees(elevation_delta), np.degrees(azimuth_delta), 0], degrees=True).as_dcm()

    # 1) rotate the source image
    from envmap import EnvironmentMap
    envmap = EnvironmentMap(erp_image_src, format_='latlong')
    newerp_image_src_new = envmap.rotate("DCM", rotation_matrix).data

    if erp_image_src.dtype == np.uint8:
        newerp_image_src_new = newerp_image_src_new.astype(np.uint8)

    return newerp_image_src_new
