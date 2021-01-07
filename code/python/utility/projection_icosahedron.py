import math
import copy
import numpy as np
from scipy import ndimage

from . import gnomonic_projection as gp
from . import spherical_coordinates as sc
from . import polygon
from . import projection

from .logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Implement icosahedron projection and stitch with the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""


def generate_icosphere_ply(mesh_file_path):
    """
    Genterate a regular icosahedron mesh.
    """
    # mesh_file_path = "icosphere.ply"
    r = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0,   r, 0.0],
        [1.0,   r, 0.0],
        [-1.0,  -r, 0.0],
        [1.0,  -r, 0.0],
        [0.0, -1.0,   r],
        [0.0,  1.0,   r],
        [0.0, -1.0,  -r],
        [0.0,  1.0,  -r],
        [r, 0.0, -1.0],
        [r, 0.0,  1.0],
        [-r, 0.0, -1.0],
        [-r, 0.0,  1.0],
    ], dtype=float)

    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [5, 4, 9],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ])

    # output the to obj file
    with open(mesh_file_path, 'w') as mesh_file:
        # output header
        mesh_file.write("ply\n")
        mesh_file.write("format ascii 1.0\n")
        mesh_file.write("element vertex {}\n".format(np.shape(vertices)[0]))
        mesh_file.write("property float x\n")
        mesh_file.write("property float y\n")
        mesh_file.write("property float z\n")
        mesh_file.write("element face {}\n".format(np.shape(faces)[0]))
        mesh_file.write("property list uchar int vertex_index\n")
        mesh_file.write("end_header\n")
        for index in range(np.shape(vertices)[0]):
            mesh_file.write("{} {} {}\n".format(vertices[index][0], vertices[index][1], vertices[index][2]))

        for index in range(np.shape(faces)[0]):
            mesh_file.write("3 {} {} {}\n".format(faces[index][0], faces[index][1], faces[index][2]))


def get_icosahedron_parameters_subdivsion(subdivision_level, triangle_index):
    """Get the parameters' of x-level subdivsion icosahedron.

    Use loop subdivision to the icosahedron.
    TODO implement  

    :param subdivision_level: The subdivsion 's level.
    :type subdivision_level: int
    :param triangle_index: The tangent triangle's index
    :type triangle_index: int
    """
    pass


def get_icosahedron_parameters(triangle_index, padding_size=0.0):
    """
    Get icosahedron's tangent face's paramters.
    Get the tangent point theta and phi. Known as the lambda_0 and phi_1.
    The erp image origin as top-left corner

    :return the tangent face's tangent point and 3 vertices's location.
    """
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # the tangent point
    lambda_0 = None
    phi_1 = None

    # the 3 points of tangent triangle in spherical coordinate
    triangle_point_00_lambda = None
    triangle_point_00_phi = None
    triangle_point_01_lambda = None
    triangle_point_01_phi = None
    triangle_point_02_lambda = None
    triangle_point_02_phi = None

    # triangles' row/col range in the erp image
    # erp_image_row_start = None
    # erp_image_row_stop = None
    # erp_image_col_start = None
    # erp_image_col_stop = None

    lambda_step = 2.0 * np.pi / 5.0
    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = -np.pi + triangle_index * lambda_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * lambda_step
        triangle_point_01_phi = np.pi / 2.0
        triangle_point_02_lambda = -np.pi + (triangle_index + 1) * lambda_step
        triangle_point_02_phi = np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = 0
        # erp_image_row_stop = (np.pi / 2 - np.arctan(0.5)) / np.pi
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp
        # erp_image_col_stop = 1.0 / 5.0 * (triangle_index_temp + 1)

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index_temp = triangle_index - 5
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index_temp * lambda_step
        phi_1 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = -np.pi + triangle_index_temp * lambda_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_lambda = -np.pi + (triangle_index_temp + 1) * lambda_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_lambda = -np.pi + lambda_step / 2.0 + triangle_index_temp * lambda_step
        triangle_point_02_phi = -np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = (np.arccos(radius_inscribed / radius_circumscribed) + np.arccos(radius_inscribed / radius_midradius)) / np.pi
        # erp_image_row_stop = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        # erp_image_col_start = 1 / 5.0 * triangle_index_temp
        # erp_image_col_stop = 1 / 5.0 * (triangle_index_temp + 1)

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index_temp = triangle_index - 10
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + triangle_index_temp * lambda_step
        phi_1 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_00_lambda = - np.pi - lambda_step / 2.0 + triangle_index_temp * lambda_step
        if triangle_index_temp == 10:
            # cross the ERP image boundary
            triangle_point_00_lambda = triangle_point_00_lambda + 2 * np.pi
        triangle_point_01_lambda = -np.pi + triangle_index_temp * lambda_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_lambda = - np.pi + lambda_step / 2.0 + triangle_index_temp * lambda_step
        triangle_point_02_phi = -np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = (np.pi / 2.0 - np.arctan(0.5)) / np.pi
        # erp_image_row_stop = (np.pi - np.arccos(radius_inscribed / radius_circumscribed) - np.arccos(radius_inscribed / radius_midradius)) / np.pi
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
        # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index_temp = triangle_index - 15
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + triangle_index_temp * lambda_step
        phi_1 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = - np.pi - lambda_step / 2.0 + triangle_index_temp * lambda_step
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_01_lambda = - np.pi + lambda_step / 2.0 + triangle_index_temp * lambda_step
        # cross the ERP image boundary
        if triangle_index_temp == 15:
            triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_02_lambda = - np.pi + triangle_index_temp * lambda_step
        triangle_point_02_phi = -np.pi / 2.0

        # # spherical coordinate (0,0) is in the center of ERP image
        # erp_image_row_start = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        # erp_image_row_stop = 1.0
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
        # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

    tangent_point = [lambda_0, phi_1]

    # the 3 points gnomonic coordinate in tangent image's gnomonic space
    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_lambda, triangle_point_00_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_lambda, triangle_point_01_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_lambda, triangle_point_02_phi, lambda_0, phi_1))

    # pading the tangent image
    triangle_points_tangent_pading = polygon.enlarge_polygon(triangle_points_tangent, padding_size)

    # if padding_size != 0.0:
    triangle_points_tangent = copy.deepcopy(triangle_points_tangent_pading)

    # the points in spherical location
    triangle_points_sph = []
    for index in range(3):
        tri_pading_x, tri_pading_y = triangle_points_tangent_pading[index]
        triangle_point_lambda, triangle_point_phi = gp.reverse_gnomonic_projection(tri_pading_x, tri_pading_y, lambda_0, phi_1)
        triangle_points_sph.append([triangle_point_lambda, triangle_point_phi])

    # compute bounding box of the face in spherical coordinate
    availied_sph_area = []
    availied_sph_area = np.array(copy.deepcopy(triangle_points_sph))
    triangle_points_tangent_pading = np.array(triangle_points_tangent_pading)
    point_insert_x = np.sort(triangle_points_tangent_pading[:, 0])[1]
    point_insert_y = np.sort(triangle_points_tangent_pading[:, 1])[1]
    availied_sph_area = np.append(availied_sph_area, [gp.reverse_gnomonic_projection(point_insert_x, point_insert_y, lambda_0, phi_1)], axis=0)
    # the bounding box of the face with spherical coordinate
    availied_ERP_area_sph = []  # [min_longitude, max_longitude, min_latitude, max_lantitude]
    availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
    availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
    if 0 <= triangle_index <= 4:
        availied_ERP_area_sph.append(np.pi / 2.0)
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down
    elif 15 <= triangle_index <= 19:
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(-np.pi / 2.0)
    else:
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))

    # else:
    #     triangle_points_sph.append([triangle_point_00_lambda, triangle_point_00_phi])
    #     triangle_points_sph.append([triangle_point_01_lambda, triangle_point_01_phi])
    #     triangle_points_sph.append([triangle_point_02_lambda, triangle_point_02_phi])

    #     availied_ERP_area.append(erp_image_row_start)
    #     availied_ERP_area.append(erp_image_row_stop)
    #     availied_ERP_area.append(erp_image_col_start)
    #     availied_ERP_area.append(erp_image_col_stop)

    return {"tangent_points": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph, "availied_ERP_area": availied_ERP_area_sph}


def erp2ico_image(erp_image, tangent_image_width, padding_size=0.0, full_face_image = False):
    """Project the equirectangular image to 20 triangle images.

    Project the equirectangular image to level-0 icosahedron.

    :param erp_image: the input equirectangular image.
    :type erp_image: numpy array, [height, width, 3]
    :param tangent_image_width: the output triangle image size, defaults to 480
    :type tangent_image_width: int, optional
    :param padding_size: the output face image' padding size
    :type padding_size: float
    :param full_face_image: If yes project all pixels in the face image, no just project the pixels in the face triangle, defaults to False
    :type full_face_image: bool, optional
    :return: a list contain 20 triangle images, the image is 4 channels, invalided pixel's alpha is 0, others is 1
    :type list
    """
    # ERP image size
    if np.shape(erp_image)[2] == 4:
        erp_image = erp_image[:, :, 0:3]
    erp_image_height = np.shape(erp_image)[0]
    erp_image_width = np.shape(erp_image)[1]

    if erp_image_width != erp_image_height * 2:
        raise Exception("the ERP image dimession is {}".format(np.shape(erp_image)))

    tangent_image_list = []
    tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)

    # generate tangent images
    for triangle_index in range(0, 20):
        log.debug("generate the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        # the face gnomonic range in tangent space
        gnomonic_x_min = np.amin(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_x_max = np.amax(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_y_min = np.amin(tangent_triangle_vertices[:, 1], axis=0)
        gnomonic_y_max = np.amax(tangent_triangle_vertices[:, 1], axis=0)
        gnom_range_x = np.linspace(gnomonic_x_min, gnomonic_x_max, num=tangent_image_width, endpoint=True)
        gnom_range_y = np.linspace(gnomonic_y_max, gnomonic_y_min, num=tangent_image_height, endpoint=True) 
        # TODO Check the order of y axis
        # gnom_range_y = np.linspace(gnomonic_y_min, gnomonic_y_max, num=tangent_image_height, endpoint=True)
        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        # the tangent triangle points coordinate in tangent image
        gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
        pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
        if not full_face_image:
            inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(gnom_range_xv.shape)

        # project to tangent image
        tangent_points = triangle_param["tangent_points"]
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2erp(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                             0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        tangent_image = np.full([tangent_image_height, tangent_image_width, 4], 255)
        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y, tangent_image_x, channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255)

        # set the pixels outside the boundary to transparent
        tangent_image[:, :, 3] = 0
        tangent_image[tangent_image_y, tangent_image_x, 3] = 255
        tangent_image_list.append(tangent_image)

    return tangent_image_list


def ico2erp_image(tangent_images, erp_image_height, padding_size=0.0):
    """Stitch the level-0 icosahedron's tangent image to ERP image.

    TODO there are seam on the stitched erp image.

    :param tangent_images: 20 tangent images in order.
    :type tangent_images: a list of numpy
    :param erp_image_height: the output erp image's height.
    :type erp_image_height: int
    :param padding_size: the face image's padding size
    :type padding_size: float
    :return: the stitched ERP image
    :type numpy
    """
    if len(tangent_images) != 20:
        log.error("The tangent's images triangle number is {}.".format(len(tangent_images)))

    images_channels_number = tangent_images[0].shape[2]
    if images_channels_number == 4:
        log.debug("the face image is RGBA image, convert the output to RGB image.")
        images_channels_number = 3
    erp_image_width = erp_image_height * 2
    erp_image = np.full([erp_image_height, erp_image_width, images_channels_number], 0)

    tangent_image_height = tangent_images[0].shape[0]
    tangent_image_width = tangent_images[0].shape[1]

    # stitch all tangnet images to ERP image
    for triangle_index in range(0, 20):
        log.debug("stitch the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        # 1) get all tangent triangle's available pixels coordinate
        availied_ERP_area = triangle_param["availied_ERP_area"]
        erp_image_col_start, erp_image_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_image_height, wrap_around=False)
        erp_image_col_stop, erp_image_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_image_height, wrap_around=False)

        # process the image boundary
        erp_image_col_start = int(erp_image_col_start) if int(erp_image_col_start) > 0 else int(erp_image_col_start - 0.5)
        erp_image_col_stop = int(erp_image_col_stop + 0.5) if int(erp_image_col_stop) > 0 else int(erp_image_col_stop)
        erp_image_row_start = int(erp_image_row_start) if int(erp_image_row_start) > 0 else int(erp_image_row_start - 0.5)
        erp_image_row_stop = int(erp_image_row_stop + 0.5) if int(erp_image_row_stop) > 0 else int(erp_image_row_stop)

        triangle_x_range = np.linspace(erp_image_col_start, erp_image_col_stop, erp_image_col_stop - erp_image_col_start + 1)
        triangle_y_range = np.linspace(erp_image_row_start, erp_image_row_stop, erp_image_row_stop - erp_image_row_start + 1)
        triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)
        # process the wrap around
        triangle_xv = np.remainder(triangle_xv, erp_image_width)
        triangle_yv = np.remainder(triangle_yv, erp_image_height)

        # 2) sample the pixel value from tanget image
        # project spherical coordinate to tangent plane
        spherical_uv = sc.erp2sph([triangle_xv, triangle_yv], erp_image_height=erp_image_height, wrap_around=False)
        lambda_0 = triangle_param["tangent_points"][0]
        phi_1 = triangle_param["tangent_points"][1]
        tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], lambda_0, phi_1)

        # the pixels in the tangent triangle
        triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])
        pixel_eps = abs(tangent_xv[0, 0] - tangent_xv[0, 1]) / (2 * tangent_image_width)
        available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                     triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

        # the tangent available gnomonic coordinate sample the pixel from the tangent image
        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                   0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        for channel in range(0, images_channels_number):
            erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] = \
                ndimage.map_coordinates(tangent_images[triangle_index][:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)

    return erp_image


def erp2ico_flow(erp_flow_mat, tangent_image_width, padding_size =  0.0, full_face_image = False):
    """Project the ERP flow to the 20 tangent flows base on the gnomonic projection.

    :param erp_flow_mat: The ERP flow image.
    :type erp_flow_mat: numpy
    :param tangent_image_width: The output tangent image width, and deduce the height by the float triangle size.
    :type tangent_image_width: int
    :param padding_size: the tangent image's padding size, defaults to 0.0
    :type padding_size: float, optional
    :param full_face_image: If yes project all pixels in the face image, no just project the pixels in the face triangle, defaults to False
    :type full_face_image: bool, optional
    :return: a list ontain 20 triangle images
    :rtype: list 
    """
    # TODO Test padding
    # get the ERP flow map parameters
    erp_image_height = np.shape(erp_flow_mat)[0]
    erp_image_width = np.shape(erp_flow_mat)[1]
    erp_flow_channel = np.shape(erp_flow_mat)[2]
    if erp_flow_channel != 2:
        log.error("The flow is not 2 channels.")

    # get the tangent image parameters
    tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)

    ico_tangent_flows = []
    for triangle_index in range(0, 20):
        log.debug("generate the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        # the face gnomonic coordinate range
        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        gnomonic_x_min = np.amin(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_x_max = np.amax(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_y_min = np.amin(tangent_triangle_vertices[:, 1], axis=0)
        gnomonic_y_max = np.amax(tangent_triangle_vertices[:, 1], axis=0)
        gnom_range_x = np.linspace(gnomonic_x_min, gnomonic_x_max, num=tangent_image_width, endpoint=True)
        gnom_range_y = np.linspace(gnomonic_y_max, gnomonic_y_min, num=tangent_image_height, endpoint=True)  # in gnomonic coordinate Y is up
        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        # the valide pixels in the tangent triangle area
        gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
        pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
        if not full_face_image:
            inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(np.shape(gnom_range_xv))

        # 0) Get the tangent image pixels' ERP location, convert the ERP optical flow's UV to tangent image's UV
        # flow start point from gnomonic --> spherical coordinate --> pixel location
        tangent_points = triangle_param["tangent_points"]  # tangent center project point
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])
        # TODO check the wrap around implement
        face_erp_pixel_x, face_erp_pixel_y = sc.sph2erp(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=False)

        # 1) comput the end point location in the tangent image
        # # get the tangent image pixels flow value in ERP image
        face_flow_pixels_src = np.zeros((face_erp_pixel_y.shape[0], erp_flow_channel), dtype=float)
        for channel in range(0, erp_flow_channel):
            face_flow_pixels_src[:, channel] = ndimage.map_coordinates(erp_flow_mat[:, :, channel], [face_erp_pixel_y, face_erp_pixel_x], order=1, mode='wrap')

        # the flow end point in ERP image pixel coordinate
        face_erp_pixel_x_target = face_erp_pixel_x + face_flow_pixels_src[:, 0]  # [inside_list]
        face_erp_pixel_y_target = face_erp_pixel_y + face_flow_pixels_src[:, 1]  # [inside_list]

        # spherical location --> tangent pixel location
        face_pixel_sph = sc.erp2sph([face_erp_pixel_x_target, face_erp_pixel_y_target], erp_image_height, wrap_around=False)
        face_image_x_target, face_image_y_target = gp.gnomonic_projection(face_pixel_sph[0, :], face_pixel_sph[1, :], tangent_points[0], tangent_points[1])

        # 2) copute the tangent image pixels optical flow
        # gnomonic coordinate -> image coordinate
        gnomonic2image_width_ratio = (tangent_image_width - 1) / (gnomonic_x_max - gnomonic_x_min)
        face_flow_u = (face_image_x_target - gnom_range_xv[inside_list]) * gnomonic2image_width_ratio
        gnomonic2image_height_ratio = (tangent_image_height - 1) / (gnomonic_y_max - gnomonic_y_min)
        face_flow_v = (face_image_y_target - gnom_range_yv[inside_list]) * gnomonic2image_height_ratio
        face_flow_v = -face_flow_v  # transform to tangent image coordinate system

        # TODO how to express the Invalid flow number?
        tangent_flow = np.full([tangent_image_height, tangent_image_width, 2], 0)
        tangent_flow[:, :, 0][inside_list] = face_flow_u
        # tangent_flow[:, :, 0][inside_list] = np.zeros(face_flow_u.shape, np.float)
        tangent_flow[:, :, 1][inside_list] = face_flow_v  # np.zeros(face_flow_u.shape, np.float)# face_flow_v
        ico_tangent_flows.append(tangent_flow)

    return ico_tangent_flows


def ico2erp_flow(tangent_flows_list, erp_flow_height=None, padding_size=0.0):
    """Stitch all 20 tangent flows to a ERP flow.

    :param tangent_flows_list: The list of 20 tangnet flow data.
    :type tangent_flows_list: list of numpy 
    :param erp_flow_height: the height of stitched ERP flow image 
    :type erp_flow_height: int
    :param padding_size: the each face's flow padding area size, defaults to 0.0
    :type padding_size: float, optional
    :return: the stitched ERP flow
    :rtype: numpy
    """
    # check the face images number
    if not 20 == len(tangent_flows_list):
        log.error("the ico face flow number is not 20")

    # get ERP and face's flow parameters
    tangent_flow_height = np.shape(tangent_flows_list[0])[0]
    tangent_flow_width = np.shape(tangent_flows_list[0])[1]
    if erp_flow_height is None:
        erp_flow_height = int(tangent_flow_height * 2.0)
    erp_flow_height = int(erp_flow_height)
    erp_flow_width = int(erp_flow_height * 2.0)
    erp_flow_channel = np.shape(tangent_flows_list[0])[2]
    if not erp_flow_channel == 2:
        log.error("The flow channels number is {}".format(erp_flow_channel))

    erp_flow_mat = np.zeros((erp_flow_height, erp_flow_width, 2), dtype=np.float64)
    erp_flow_weight_mat = np.zeros((erp_flow_height, erp_flow_width), dtype=np.float64)

    for triangle_index in range(0, len(tangent_flows_list)):
        log.debug("stitch the tangent image {}".format(triangle_index))

        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)
        lambda_0 = triangle_param["tangent_points"][0]
        phi_1 = triangle_param["tangent_points"][1]
        triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])

        # 1) get all tangent triangle's available pixels coordinate
        # availed pixles range in ERP spherical coordinate
        availied_ERP_area = triangle_param["availied_ERP_area"]
        erp_flow_col_start, erp_flow_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_flow_height, wrap_around=False)
        erp_flow_col_stop, erp_flow_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_flow_height, wrap_around=False)

        # 2) get the pixels location in tangent image location
        # process the tangent flow boundary
        erp_flow_col_start = int(erp_flow_col_start) if int(erp_flow_col_start) > 0 else int(erp_flow_col_start - 0.5)
        erp_flow_col_stop = int(erp_flow_col_stop + 0.5) if int(erp_flow_col_stop) > 0 else int(erp_flow_col_stop)
        erp_flow_row_start = int(erp_flow_row_start) if int(erp_flow_row_start) > 0 else int(erp_flow_row_start - 0.5)
        erp_flow_row_stop = int(erp_flow_row_stop + 0.5) if int(erp_flow_row_stop) > 0 else int(erp_flow_row_stop)
        triangle_x_range = np.linspace(erp_flow_col_start, erp_flow_col_stop, erp_flow_col_stop - erp_flow_col_start + 1)
        triangle_y_range = np.linspace(erp_flow_row_start, erp_flow_row_stop, erp_flow_row_stop - erp_flow_row_start + 1)
        triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)
        triangle_xv = np.remainder(triangle_xv, erp_flow_width)  # process the wrap around
        triangle_yv = np.remainder(triangle_yv, erp_flow_height)
        # ERP image space --> spherical space
        spherical_uv = sc.erp2sph((triangle_xv, triangle_yv), erp_flow_height, False)

        # spherical space --> normailzed tangent image space
        tangent_xv_gnom, tangent_yv_gnom = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], lambda_0, phi_1)

        # the available (in the triangle) pixels list
        pixel_eps = abs(tangent_xv_gnom[0, 0] - tangent_xv_gnom[0, 1]) / (2 * tangent_flow_width)
        available_list = gp.inside_polygon_2d(np.stack((tangent_xv_gnom.flatten(), tangent_yv_gnom.flatten()), axis=1), triangle_points_tangent, on_line=True, eps=pixel_eps)
        available_list = available_list.reshape(tangent_xv_gnom.shape)

        # normailzed tangent image space --> tangent image space
        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_xv_pixel, tangent_yv_pixel = gp.gnomonic2pixel(tangent_xv_gnom[available_list], tangent_yv_gnom[available_list], 0.0, tangent_flow_width, tangent_flow_height, tangent_gnomonic_range)

        # 3) get the value of interpollations
        # 3-0) remove the pixels outside the tangent image
        # get the tangent images flow in the tangent image space
        face_flow_x = ndimage.map_coordinates(tangent_flows_list[triangle_index][:, :, 0],[tangent_yv_pixel, tangent_xv_pixel], order=1, mode='constant', cval=255)
        tangent_xv_pixel_tar_available = tangent_xv_pixel + face_flow_x

        face_flow_y = ndimage.map_coordinates(tangent_flows_list[triangle_index][:, :, 1], [tangent_yv_pixel, tangent_xv_pixel], order=1, mode='constant', cval=255)
        tangent_yv_pixel_tar_available = tangent_yv_pixel + face_flow_y

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        tangent_xv_tar_gnom, tangent_yv_tar_gnom = gp.pixel2gnomonic(tangent_xv_pixel_tar_available, tangent_yv_pixel_tar_available, 0.0, tangent_flow_width, tangent_flow_height, tangent_gnomonic_range)
        # tangent normailzed space --> spherical space
        tangent_phi_tar_sph, tangent_theta_tar_sph = gp.reverse_gnomonic_projection(tangent_xv_tar_gnom, tangent_yv_tar_gnom, lambda_0, phi_1)
        # spherical space --> ERP image space
        tangent_xv_tar_pixel, tangent_yv_tar_pixel = sc.sph2erp(tangent_phi_tar_sph, tangent_theta_tar_sph, erp_flow_height, True)


        # 4) get ERP flow with source and target pixels location
        # 4-0) the ERP flow
        face_flow_u = tangent_xv_tar_pixel - triangle_xv[available_list]
        face_flow_v = tangent_yv_tar_pixel - triangle_yv[available_list]

        # 4-1) TODO blend the optical flow
        # # comput the all available pixels' weight
        # weight_type = "normal_distribution_flowcenter"
        # face_weight_mat_1 = projection.get_blend_weight(face_x_src_gnomonic[available_list].flatten(), face_y_src_gnomonic[available_list].flatten(), weight_type, np.stack((face_flow_x, face_flow_y), axis=1))
        # weight_type = "image_warp_error"
        # face_weight_mat_2 = projection.get_blend_weight(face_erp_x[available_list], face_erp_y[available_list], weight_type, np.stack((face_x_tar_available, face_y_tar_available), axis=1), image_erp_src, image_erp_tar)
        # face_weight_mat = np.multiply(face_weight_mat_1, face_weight_mat_2)

        # face_weight_mat = np.ones(tangent_yv_pixel.shape, dtype= np.float64)
        # face_weight_mat[available_list] = 1
        # # for debug weight
        # if not flow_index == -1:
        #     from . import image_io
        #     temp = np.zeros(face_x_src_gnomonic.shape, np.float)
        #     temp[available_list] = face_weight_mat
        #     image_io.image_show(temp)

        erp_flow_mat[triangle_yv[available_list].astype(np.int64), triangle_xv[available_list].astype(np.int64), 0] = face_flow_u  # * face_weight_mat
        erp_flow_mat[triangle_yv[available_list].astype(np.int64), triangle_xv[available_list].astype(np.int64), 1] = face_flow_v  # * face_weight_mat
        # erp_flow_weight_mat[ triangle_yv[available_list].astype(np.int64), triangle_xv[available_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), np.float) # debug
    # non_zero_weight_list = erp_flow_weight_mat != 0
    # if not np.all(non_zero_weight_list):
    #     log.warn("the optical flow weight matrix contain 0.")
    # for channel_index in range(0, 2):
    #     erp_flow_mat[:, :, channel_index][non_zero_weight_list] = erp_flow_mat[:, :, channel_index][non_zero_weight_list] / erp_flow_weight_mat[non_zero_weight_list]

    return erp_flow_mat
