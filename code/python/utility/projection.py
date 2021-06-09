import numpy as np
from scipy import ndimage
from scipy.stats import norm

import gnomonic_projection
import spherical_coordinates as sc

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def get_blend_weight_ico(face_x_src_gnomonic, face_y_src_gnomonic,
                         weight_type,
                         flow_uv=None,
                         image_erp_src=None, image_erp_tar=None,
                         gnomonic_bounding_box=None):
    """Compute the faces's weight.

    :param face_x_src_gnomonic: the pixel's gnomonic coordinate x in tangent image
    :type face_x_src_gnomonic: numpy, [pixel_number]
    :param face_y_src_gnomonic: the pixel's gnomonic coordinate y in tangent image
    :type face_y_src_gnomonic: numpy, [pixel_number]
    :param weight_type: The weight compute method, [straightforward|cartesian_distance]
    :type weight_type: str
    :param flow_uv: the tangent face forward optical flow which is in image coordinate, unit is pixel.
    :type flow_uv: numpy 
    :param image_erp_src: the source ERP rgb image, used to compute the optical flow warp error
    :type: numpy
    :param image_erp_tar: the target ERP rgb image used to compute the optical flow warp error
    :type: numpy
    :param gnomonic_bounding_box: the available pixels area's bounding box
    :type: list
    :return: the cubemap's face weight used to blend different faces to ERP image.
    :rtype: numpy
    """
    weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)

    if weight_type == "straightforward":
        available_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box, on_line=False, eps=1e-7)
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


def get_blend_weight_cubemap(face_x_src_gnomonic, face_y_src_gnomonic,
                             weight_type,
                             flow_uv=None,
                             image_erp_src=None, image_erp_tar=None,
                             gnomonic_bounding_box=None):
    """Compute the faces's weight.

    :param face_x_src_gnomonic: the pixel's gnomonic coordinate x in tangent image
    :type face_x_src_gnomonic: numpy, [pixel_number]
    :param face_y_src_gnomonic: the pixel's gnomonic coordinate y in tangent image
    :type face_y_src_gnomonic: numpy, [pixel_number]
    :param weight_type: The weight compute method, [straightforward|cartesian_distance]
    :type weight_type: str
    :param flow_uv: the tangent face forward optical flow which is in image coordinate, unit is pixel.
    :type flow_uv: numpy 
    :param image_erp_src: the source ERP rgb image, used to compute the optical flow warp error
    :type: numpy
    :param image_erp_tar: the target ERP rgb image used to compute the optical flow warp error
    :type: numpy
    :param gnomonic_bounding_box: the available pixels area's bounding box
    :type: list
    :return: the cubemap's face weight used to blend different faces to ERP image.
    :rtype: numpy
    """
    weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)

    if weight_type == "straightforward":
        # just set the pixels in this cube map face range is available. [-1, +1, -1, +1]
        if gnomonic_bounding_box is None:
            pbc = 1
            gnomonic_bounding_box = np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]])
        available_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box, on_line=True, eps=1e-7)
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


def flow2sph_rotation(erp_flow, use_weight=True):
    """Compute the  two image rotation from the ERP image's optical flow.
    The rotation is from the first image to second image.

    :param erp_flow: the erp image's flow 
    :type erp_flow: numpy 
    :param use_weight: use the centre rows and columns to compute the rotation, default is True.
    :type: bool
    :return: the offset of ERP image, [theta shift, phi shift
    :rtype: float
    """
    erp_image_height = erp_flow.shape[0]
    erp_image_width = erp_flow.shape[1]

    # convert the pixel offset to rotation radian
    theta_delta_array = 2.0 * np.pi * (erp_flow[:, :, 0] / erp_image_width)
    phi_delta_array = np.pi * (erp_flow[:, :, 1] / erp_image_height)

    if use_weight:
        # weight of the u, width
        stdev = erp_image_height * 0.5 * 0.25
        weight_u_array_index = np.arange(erp_image_height)
        weight_u_array = norm.pdf(weight_u_array_index, erp_image_height / 2.0, stdev)
        theta_delta_array = np.average(theta_delta_array, axis=0, weights=weight_u_array)

        # weight of the v, height
        stdev = erp_image_width * 0.5 * 0.25
        weight_v_array_index = np.arange(erp_image_width)
        weight_v_array = norm.pdf(weight_v_array_index, erp_image_width / 2.0, stdev)
        phi_delta_array = np.average(phi_delta_array, axis=1,  weights=weight_v_array)

    theta_delta = np.mean(theta_delta_array)
    phi_delta = np.mean(phi_delta_array)

    return theta_delta, phi_delta


def image_rotate_flow(erp_image, erp_flow):
    """
    Rotate the ERP image base on the flow. 
    The flow is from erp_image to another image.

    :param erp_image: the flow's ERP image, the image is 
    :type erp_image: numpy 
    :param erp_flow: the erp image's flow.
    :type erp_flow: numpy 
    :return: The rotated ERP image
    :rtype: numpy
    """
    # compuate the average of optical flow & get the delta theta and phi
    # TODO check the flow overflow or not
    theta_delta, phi_delta = flow2sph_rotation(erp_flow)
    # rotate the ERP image
    erp_image_rot = sc.rotate_erp_array(erp_image, theta_delta, phi_delta)
    if erp_image.dtype == np.uint8:
        erp_image_rot = erp_image_rot.astype(np.uint8)
    return erp_image_rot, theta_delta, phi_delta


def flow_rotate_endpoint(optical_flow, rotation):
    """ Add the rotation offset to the end points of optical flow.

    :param optical_flow: the original optical flow, [height, width, 2]
    :type optical_flow: numpy
    :param rotation: the rotation of spherical coordinate in radian, [theta, phi]
    :type rotation: tuple
    :return: the new optical flow
    :rtype: numpy 
    """
    flow_height = optical_flow.shape[0]
    flow_width = optical_flow.shape[1]
    end_points_array_x = np.linspace(0, flow_width, flow_width, endpoint=False)
    end_points_array_y = np.linspace(0, flow_height, flow_height, endpoint=False)
    src_points_array_xv, src_points_array_yv = np.meshgrid(end_points_array_x, end_points_array_y)

    # get end point location in ERP coordinate
    end_points_array_xv = np.remainder(src_points_array_xv + optical_flow[:, :, 0], flow_width)
    end_points_array_yv = np.remainder(src_points_array_yv + optical_flow[:, :, 1], flow_height)
    end_points_array = sc.rotation2erp_motion_vector((flow_height, flow_width),  rotation[0],  rotation[1])

    end_points_array_xv += ndimage.map_coordinates(end_points_array[:, :, 0], [end_points_array_yv, end_points_array_xv], order=1, mode='constant', cval=255)
    end_points_array_yv += ndimage.map_coordinates(end_points_array[:, :, 1], [end_points_array_yv, end_points_array_xv], order=1, mode='constant', cval=255)
    end_points_array_xv = np.remainder(end_points_array_xv, flow_width)
    end_points_array_yv = np.remainder(end_points_array_yv, flow_height)   

    # end_points_array_xv = sc.rotate_erp_array(end_points_array_xv,  rotation[0],  rotation[1])
    # end_points_array_yv = sc.rotate_erp_array(end_points_array_yv,  rotation[0],  rotation[1])
    # # rotation the target points
    # phi_delta= rotation[1]
    # theta_delta = rotation[0]
    # from scipy.spatial.transform import Rotation as R
    # rotation_matrix = R.from_euler("xyz", [np.degrees(phi_delta), np.degrees(theta_delta), 0], degrees=True).as_dcm()
    # from envmap import EnvironmentMap
    # end_points_array_xv = EnvironmentMap(end_points_array_xv[:,:,np.newaxis], format_='latlong').rotate("DCM", rotation_matrix).data[:,:,0]
    # end_points_array_yv = EnvironmentMap(end_points_array_yv[:,:,np.newaxis], format_='latlong').rotate("DCM", rotation_matrix).data[:,:,0]
    # # erp -> spherical
    # end_points_sph = sc.erp2sph([end_points_array_xv, end_points_array_yv], erp_image_height=flow_height)
    # end_points_sph[0, :, :] += rotation[0]
    # end_points_sph[1, :, :] += rotation[1]  # TODO correct
    # # spherical -> epr
    # end_points_erp_accu_x, end_points_erp_accu_y = sc.sph2erp(end_points_sph[0, :, :], end_points_sph[1, :, :], flow_height)

    # erp pixles location to flow
    end_points_array_xv -= src_points_array_xv
    end_points_array_yv -= src_points_array_yv
    return np.stack((end_points_array_xv, end_points_array_yv), axis=-1)
