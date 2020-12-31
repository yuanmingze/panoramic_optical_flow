import math
import copy
import numpy as np
from scipy import ndimage

from . import gnomonic_projection as gp
from . import spherical_coordinates as sc
from . import polygon

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
    availied_ERP_area = [] # [min_longitude, max_longitude, min_latitude, max_lantitude]
    availied_ERP_area.append(np.amin(availied_sph_area[:, 0]))
    availied_ERP_area.append(np.amax(availied_sph_area[:, 0]))
    if 0 <= triangle_index <= 4:
        availied_ERP_area.append(np.pi / 2.0)
        availied_ERP_area.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down
    elif 15 <= triangle_index <= 19:
        availied_ERP_area.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area.append(-np.pi / 2.0)
    else:
        availied_ERP_area.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area.append(np.amin(availied_sph_area[:, 1]))

    # else:
    #     triangle_points_sph.append([triangle_point_00_lambda, triangle_point_00_phi])
    #     triangle_points_sph.append([triangle_point_01_lambda, triangle_point_01_phi])
    #     triangle_points_sph.append([triangle_point_02_lambda, triangle_point_02_phi])

    #     availied_ERP_area.append(erp_image_row_start)
    #     availied_ERP_area.append(erp_image_row_stop)
    #     availied_ERP_area.append(erp_image_col_start)
    #     availied_ERP_area.append(erp_image_col_stop)

    return {"tangent_points": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph, "availied_ERP_area": availied_ERP_area}


def erp2ico_image(erp_image, tangent_image_size, padding_size=0.0):
    """Project the equirectangular image to 20 triangle images.

    Project the equirectangular image to level-0 icosahedron.

    :param erp_image: the input equirectangular image.
    :type erp_image: numpy array, [height, width, 3]
    :param tangent_image_size: the output triangle image size, defaults to 480
    :type tangent_image_size: int, optional
    :param padding_size: the output face image' padding size
    :type padding_size: float
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

    tangent_image_width = tangent_image_size
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
        gnom_range_y = np.linspace(gnomonic_y_min, gnomonic_y_max, num=tangent_image_height, endpoint=True)
        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        tangent_image = np.full([tangent_image_height, tangent_image_width, 4], 255)
        # the tangent triangle points coordinate in tangent image
        gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
        pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
        inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
        inside_list = inside_list.reshape(np.shape(gnom_range_xv))

        # project to tangent image
        tangent_points = triangle_param["tangent_points"]
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2epr(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                             0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

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
        erp_image_col_start, erp_image_row_start = sc.sph2epr(availied_ERP_area[0], availied_ERP_area[2], erp_image_height, wrap_around=False)
        erp_image_col_stop, erp_image_row_stop = sc.sph2epr(availied_ERP_area[1], availied_ERP_area[3], erp_image_height, wrap_around=False)

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


def erp2ico_flow(erp_flow_mat, tangent_image_width, padding_size):
    """Project the ERP flow to the 20 tangent flows base on the inverse gnomonic projection.

    :param erp_flow_mat: The ERP flow image.
    :type erp_flow_mat: numpy
    :param tangent_image_width: The output tangent image width, and deduce the height by the float triangle size.
    :type tangent_image_width: int
    :param padding_size: the tangent image's padding size, defaults to 0.0
    :type padding_size: float, optional
    :return: a list ontain 20 triangle images
    :rtype: list 
    """
    # get the ERP flow map parameters
    erp_image_height = np.shape(erp_flow_mat)[0]
    erp_image_width = np.shape(erp_flow_mat)[1]
    erp_flow_channel = np.shape(erp_flow_mat)[2]
    if erp_flow_channel != 2:
        log.error("The flow is not 2 channels.")

    # get the tangent image parameters
    tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)
    gnomonic2image_width_ratio = (tangent_image_width - 1) / (2.0 + padding_size * 2.0)
    gnomonic2image_height_ratio= (tangent_image_height - 1) / (2.0 + padding_size * 2.0)
    pbc = 1.0 + padding_size  # projection_boundary_coefficient

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
        # TODO in gnomonic coordinate Y is to down
        gnom_range_y = np.linspace(gnomonic_y_max,gnomonic_y_min, num=tangent_image_height, endpoint=True) 
        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        # the tangent triangle points coordinate in tangent image
        gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
        pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
        inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
        inside_list = inside_list.reshape(np.shape(gnom_range_xv))

        # import image_io
        # image_io.image_show(inside_list)

        # 0) Get the location of tangent image's pixels corresponding location in ERP
        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        # the x,y of tangent image
        # project to tangent image
        # x_grid = np.linspace(-pbc, pbc, tangent_image_width)
        # y_grid = np.linspace(pbc, - pbc, tangent_image_height)
        # x, y = np.meshgrid(x_grid, y_grid)

        # tangent center project point
        tangent_points = triangle_param["tangent_points"]
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])

        # spherical coordinate --> pixel location
        # erp_pixel_x = ((lambda_ + np.pi) / (2 * np.pi)) * erp_image_width
        # erp_pixel_y = (- phi_ + 0.5 * np.pi) / np.pi * erp_image_height
        # sp
        # tansform from spherical coordinate --> pixel location
        # TODO check the wrap around implement
        face_erp_pixel_x, face_erp_pixel_y = sc.sph2epr(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # import ipdb; ipdb.set_trace()

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list], 0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        # interpollation
        face_flow_pixels_src = np.zeros((tangent_image_height, tangent_image_width, erp_flow_channel), dtype=float)
        for channel in range(0, erp_flow_channel):
            face_flow_pixels_src[tangent_image_y, tangent_image_x, channel] = ndimage.map_coordinates(erp_flow_mat[:, :, channel], [face_erp_pixel_y, face_erp_pixel_x], order=1, mode='wrap')

        # 1) comput the end point location in the tangent image
        # convert the ERP optical flow's UV to tangent image's UV
        face_erp_pixel_x_target = face_erp_pixel_x + face_flow_pixels_src[:, :, 0][inside_list]
        face_erp_pixel_y_target = face_erp_pixel_y + face_flow_pixels_src[:, :, 1][inside_list]


        # # process warp around
        # erp_pixel_x_target[erp_pixel_x_target < 0] = erp_pixel_x_target[erp_pixel_x_target < 0] + erp_image_width
        # erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] = erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] - erp_image_width
        # erp_pixel_y_target[erp_pixel_y_target < 0] = erp_pixel_y_target[erp_pixel_y_target < 0] + erp_image_height
        # erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] = erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] - erp_image_height
        # # convert the erp location to spherical coordinate location
        # lambda_target = erp_pixel_x_target / erp_image_width * np.pi * 2 - np.pi
        # phi_target = -erp_pixel_y_target / erp_image_height * np.pi + 0.5 * np.pi


        face_pixel_sph = sc.erp2sph([face_erp_pixel_x_target, face_erp_pixel_y_target], erp_image_height=erp_image_height, wrap_around=False)


        # spherical location to tangent location
        face_image_x_target, face_image_y_target = gp.gnomonic_projection(face_pixel_sph[0, :], face_pixel_sph[1, :], tangent_points[0], tangent_points[1])


        face_flow_u = (face_image_x_target - gnom_range_xv[inside_list]) * gnomonic2image_width_ratio
        face_flow_v = (face_image_y_target - gnom_range_yv[inside_list]) * gnomonic2image_height_ratio
        face_flow_v = -face_flow_v  # transform to image coordinate system (+y is to down)

        # temp = np.zeros((tangent_image_height, tangent_image_width), dtype=float)
        # temp[inside_list] = face_pixel_sph[1, :]
        # import image_io
        # image_io.image_show(temp)
        # import ipdb; ipdb.set_trace()
        # 2) the optical flow of tangent image
        # TODO the Invalid flow number?
        tangent_flow = np.full([tangent_image_height, tangent_image_width, 2], 0)
        tangent_flow[:, :, 0][inside_list] = face_flow_u
        tangent_flow[:, :, 1][inside_list] = face_flow_v
        # import ipdb; ipdb.set_trace()

        # face_flow = np.stack((face_flow_u, face_flow_v), axis=2)
        ico_tangent_flows.append(tangent_flow)

    return ico_tangent_flows


def ico2erp_flow(tangent_flows, erp_image_height):
    """Stitch all 20 tangent flows to a ERP flow.

    :param tangent_flows: The list of 20 tangnet flow data.
    :type tangent_flows: list of numpy 
    :param erp_image_height: the height of stitched ERP flow image 
    :type erp_image_height: int
    :return: the stitched ERP flow
    :type numpy
    """
    # TODO implement
    pass
