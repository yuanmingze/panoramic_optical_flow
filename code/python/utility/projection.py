import numpy as np
from scipy import ndimage

import flow_postproc
import polygon
import spherical_coordinates as sc
import projection_icosahedron as proj_ico
from scipy.spatial.transform import Rotation as R

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
        available_list = polygon.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box, on_line=False, eps=1e-7)
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
        # use the optical flow average to compute the center point
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
        image_erp_tar_image_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_src_image_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_warp_diff = np.zeros((pixels_number, channel_number), np.float)
        for channel in range(0, channel_number):
            image_erp_tar_image_flow[:, channel] = ndimage.map_coordinates(image_erp_tar[:, :, channel], [flow_uv[:, 1], flow_uv[:, 0]], order=1, mode='constant', cval=255)
            image_erp_src_image_flow[:, channel] = ndimage.map_coordinates(image_erp_src[:, :, channel], [face_y_src_gnomonic, face_x_src_gnomonic], order=1, mode='constant', cval=255)
            image_erp_warp_diff[:, channel] = np.absolute(image_erp_tar_image_flow[:, channel] - image_erp_src_image_flow[:, channel])

        image_erp_warp_diff = np.mean(image_erp_warp_diff, axis=1) / np.mean(image_erp_warp_diff) #255.0
        weight_map = np.exp(-image_erp_warp_diff)
        # rgb_diff = np.linalg.norm(image_erp_warp_diff, axis=1)
        # weight_map = rgb_diff
        # non_zeros_index = rgb_diff != 0.0
        # weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)
        # weight_map[non_zeros_index] = 0.95 / rgb_diff[non_zeros_index]
        # weight_map[non_zeros_index] =  rgb_diff[non_zeros_index]
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
        available_list = polygon.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box, on_line=True, eps=1e-7)
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


def flow_rotate_endpoint(optical_flow, rotation, wraparound = False):
    """ Add the rotation offset to the end points of optical flow.

    :param optical_flow: the original optical flow, [height, width, 2]
    :type optical_flow: numpy
    :param rotation: the rotation of spherical coordinate in radian, [theta, phi] or rotation matrix.
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
    # end_points_array_xv = np.remainder(src_points_array_xv + optical_flow[:, :, 0], flow_width)
    # end_points_array_yv = np.remainder(src_points_array_yv + optical_flow[:, :, 1], flow_height)
    end_points_array_xv, end_points_array_yv = flow_postproc.erp_pixles_modulo(src_points_array_xv + optical_flow[:, :, 0], src_points_array_yv + optical_flow[:, :, 1], flow_width, flow_height)

    end_points_array = None
    if isinstance(rotation, (list, tuple)):
        rotation_mat = sc.rot_sph2mat(rotation[0], rotation[1])
        end_points_array = sc.rotation2erp_motion_vector((flow_height, flow_width), rotation_mat, wraparound=True)
    elif isinstance(rotation, np.ndarray):
        end_points_array = sc.rotation2erp_motion_vector((flow_height, flow_width), rotation_matrix=rotation,wraparound=True)
    else:
        log.error("Do not support rotation data type {}.".format(type(rotation)))

    # flow_vis.flow_value_to_color(end_points_array)
    rotation_flow_u = ndimage.map_coordinates(end_points_array[:, :, 0], [end_points_array_yv, end_points_array_xv], order=1, mode='wrap')
    rotation_flow_v = ndimage.map_coordinates(end_points_array[:, :, 1], [end_points_array_yv, end_points_array_xv], order=1, mode='wrap')

    # end_points_array_xv = np.remainder(end_points_array_xv + rotation_flow_u, flow_width)
    # end_points_array_yv = np.remainder(end_points_array_yv + rotation_flow_v, flow_height)
    end_points_array_xv, end_points_array_yv = flow_postproc.erp_pixles_modulo(end_points_array_xv + rotation_flow_u, end_points_array_yv + rotation_flow_v, flow_width, flow_height)

    # erp pixles location to flow
    end_points_array_xv -= src_points_array_xv
    end_points_array_yv -= src_points_array_yv
    flow_rotated = np.stack((end_points_array_xv, end_points_array_yv), axis=-1)
    if wraparound:
        flow_rotated = flow_postproc.erp_of_wraparound(flow_rotated)
    return  flow_rotated


def ico_padding2fov(padding_size = 0.0):
    """
    Convert the ico projection with padding to FoV parameters.

    """


    pass

def cube_padding2fov(padding_size = 0.0):
    """

    """

    pass




def tangent_image_resolution(erp_image_width, padding_size):
    """Get the suggested tangent image resolution base on the FoV.

    :param erp_image_width: [description]
    :type erp_image_width: [type]
    :param padding_size: [description]
    :type padding_size: [type]
    :return: recommended tangent image size in pixel.
    :rtype: int
    """
    # camera intrinsic parameters
    ico_param_list = proj_ico.get_icosahedron_parameters(7, padding_size)
    triangle_points_tangent = ico_param_list["triangle_points_tangent"]
    # compute the tangent image resoution.
    tangent_points_x_min = np.amin(np.array(triangle_points_tangent)[:, 0])
    fov_h = np.abs(2 * np.arctan2(tangent_points_x_min, 1.0))
    tangent_image_width = erp_image_width * (fov_h / (2 * np.pi))
    tangent_image_height = 0.5 * tangent_image_width / np.tan(np.radians(30.0))
    return int(tangent_image_width + 0.5), int(tangent_image_height + 0.5)


def ico_projection_cam_params(image_width = 400, padding_size=0):
    """    
    Figure out the camera intrinsic parameters for 20 faces of icosahedron.
    It does not need camera parameters.

    :param image_width: Tangent image's width, the image height derive from image ratio.
    :type image_width: int
    :param padding_size: The tangent face padding size, defaults to 0
    :type padding_size: float, optional
    :return: 20 faces camera parameters.
    :rtype: list
    """
    # camera intrinsic parameters
    ico_param_list = proj_ico.get_icosahedron_parameters(7, padding_size)
    tangent_point = ico_param_list["tangent_point"]
    triangle_points_tangent = ico_param_list["triangle_points_tangent"]

    # use tangent plane
    tangent_points_x_min = np.amin(np.array(triangle_points_tangent)[:, 0])
    tangent_points_y_min = np.amin(np.array(triangle_points_tangent)[:, 1])
    tangent_points_y_max = np.amax(np.array(triangle_points_tangent)[:, 1])
    fov_v = np.abs(np.arctan2(tangent_points_y_min, 1.0)) + np.abs(np.arctan2(tangent_points_y_max, 1.0))
    fov_h = np.abs(2 * np.arctan2(tangent_points_x_min, 1.0))

    log.debug("Pin-hole camera fov_h: {}, fov_v: {}".format(np.degrees(fov_h), np.degrees(fov_v)))

    # image aspect ratio, the triangle is equilateral triangle
    image_height = 0.5 * image_width / np.tan(np.radians(30.0))
    fx = 0.5 * image_width / np.tan(fov_h * 0.5)
    fy = 0.5 * image_height / np.tan(fov_v * 0.5)

    cx = (image_width - 1) / 2.0
    # invert and upright triangle cy
    cy_invert = 0.5 * (image_width - 1.0) * np.tan(np.radians(30.0)) + 10.0
    cy_up = 0.5 * (image_width - 1.0) / np.sin(np.radians(60.0)) + 10.0

    subimage_cam_param_list = []
    for index in range(0, 20):
        # intrinsic parameters
        cy = None
        if 0 <= index <= 4:
            cy = cy_up
        elif 5 <= index <= 9:
            cy = cy_invert
        elif 10 <= index <= 14:
            cy = cy_up
        else:
            cy = cy_invert

        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]])

        # rotation
        ico_param_list = proj_ico.get_icosahedron_parameters(index, padding_size)
        tangent_point = ico_param_list["tangent_point"]
        # print(tangent_point)
        rot_y = tangent_point[0]
        rot_x = tangent_point[1]
        rotation = R.from_euler("zyx", [0.0, -rot_y, -rot_x], degrees=False)
        rotation_mat = rotation.as_matrix()

        params = {'rotation': rotation_mat,
                  'translation': np.array([0, 0, 0]),
                  "fov_h" : fov_h,
                  "fov_v": fov_v,
                  'intrinsic': {
                      'image_width': image_width,
                      'image_height': image_height,
                      'focal_length_x': fx,
                      'focal_length_y': fy,
                      'principal_point': [cx, cy],
                      'matrix': intrinsic_matrix}
                  }

        subimage_cam_param_list.append(params)

    return subimage_cam_param_list
