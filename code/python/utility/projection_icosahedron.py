import math
import copy
import numpy as np
from scipy import ndimage


from . import image_io
from . import nfov
from . import gnomonic_projection as gp
from . import spherical_coordinates as sc


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

    TODO check the points order, need clockwise!
    TODO implement padding

    :return the tangent face's tangent point and 3 vertices's location.
    """

    radius_circumscribed = np.sin(2 * np.pi / 5.0)  # the
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # the tangent point
    lambda_0 = None
    phi_1 = None

    # the 3 points of tangent triangle
    triangle_point_00_lambda = None
    triangle_point_00_phi = None
    triangle_point_01_lambda = None
    triangle_point_01_phi = None
    triangle_point_02_lambda = None
    triangle_point_02_phi = None

    # triangles' row/col range in the erp image 
    erp_image_row_start = None
    erp_image_row_stop = None
    erp_image_col_start = None
    erp_image_col_stop = None

    lambda_step = 2.0 * np.pi / 5.0
    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * lambda_step
        triangle_point_00_phi = np.pi / 2.0
        triangle_point_01_lambda = -np.pi + triangle_index * lambda_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_lambda = -np.pi + (triangle_index + 1) * lambda_step
        triangle_point_02_phi = np.arctan(0.5)

        # availied area of ERP image
        erp_image_row_start = 0
        erp_image_row_stop = (np.pi / 2 - np.arctan(0.5)) / np.pi
        erp_image_col_start = 1.0 / 5.0 * triangle_index
        erp_image_col_stop = 1.0 / 5.0 * (triangle_index + 1)

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index = triangle_index - 5
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = -np.pi + triangle_index * lambda_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_lambda = -np.pi + (triangle_index + 1) * lambda_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_lambda = -np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        triangle_point_02_phi = -np.arctan(0.5)

        # availied area of ERP image
        erp_image_row_start = (np.arccos(radius_inscribed / radius_circumscribed) + np.arccos(radius_inscribed / radius_midradius)) / np.pi
        erp_image_row_stop = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        erp_image_col_start = 1 / 5.0 * triangle_index
        erp_image_col_stop = 1 / 5.0 * (triangle_index + 1)

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index = triangle_index - 10
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = -np.pi + triangle_index * lambda_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_01_lambda = - np.pi - lambda_step / 2.0 + triangle_index * lambda_step
        if triangle_index == 10:
            # cross the ERP image boundary
            triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
        triangle_point_02_lambda = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        triangle_point_02_phi = -np.arctan(0.5)

        # availied area of ERP image
        erp_image_row_start = (np.pi / 2.0 - np.arctan(0.5)) / np.pi
        erp_image_row_stop = (np.pi - np.arccos(radius_inscribed / radius_circumscribed) - np.arccos(radius_inscribed / radius_midradius)) / np.pi
        erp_image_col_start = 1.0 / 5.0 * triangle_index - 1.0 / 5.0 / 2.0
        erp_image_col_stop = 1.0 / 5.0 * triangle_index + 1.0 / 5.0 / 2.0

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index = triangle_index - 15
        # tangent point of inscribed spheric
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_lambda = - np.pi - lambda_step / 2.0 + triangle_index * lambda_step
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_01_lambda = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        # cross the ERP image boundary
        if triangle_index == 15:
            triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_02_lambda = - np.pi + triangle_index * lambda_step
        triangle_point_02_phi = -np.pi / 2.0

        # spherical coordinate (0,0) is in the center of ERP image
        erp_image_row_start = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        erp_image_row_stop = 1.0
        erp_image_col_start = 1.0 / 5.0 * triangle_index - 1.0 / 5.0 / 2.0
        erp_image_col_stop = 1.0 / 5.0 * triangle_index + 1.0 / 5.0 / 2.0

    tangent_point = [lambda_0, phi_1]

    triangle_points_sph = []
    triangle_points_sph.append([triangle_point_00_lambda, triangle_point_00_phi])
    triangle_points_sph.append([triangle_point_01_lambda, triangle_point_01_phi])
    triangle_points_sph.append([triangle_point_02_lambda, triangle_point_02_phi])

    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_lambda, triangle_point_00_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_lambda, triangle_point_01_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_lambda, triangle_point_02_phi, lambda_0, phi_1))

    availied_ERP_area = []
    availied_ERP_area.append(erp_image_row_start)
    availied_ERP_area.append(erp_image_row_stop)
    availied_ERP_area.append(erp_image_col_start)
    availied_ERP_area.append(erp_image_col_stop)

    return {"tangent_points": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph, "availied_ERP_area": availied_ERP_area}


def erp2ico_image(erp_image, tangent_image_size, padding_size = 0.0):
    """Project the equirectangular image to 20 triangle images.

    Project the equirectangular image to level-0 icosahedron.

    :param erp_image: the input equirectangular image.
    :type erp_image_path: numpy array, [height, width, 3]
    :param tangent_image_size: the output triangle image size, defaults to 480
    :type tangent_image_size: int, optional
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

    # the gnomonic projection range
    gnom_range_x = np.linspace(-1, 1, num=tangent_image_size, endpoint=True)
    gnom_range_y = np.linspace(-1, 1, num=tangent_image_size, endpoint=True)      
    gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

    # generate tangent images
    for triangle_index in range(0, 20):         
        log.debug("generate the tangent image {}".format(triangle_index))
        tangent_image = np.full([tangent_image_size, tangent_image_size, 4], 255)
        triangle_param = get_icosahedron_parameters(triangle_index)

        # the tangent triangle points coordinate in tangent image
        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        inside_list = gp.inside_polygon_2d(np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1), tangent_triangle_vertices).reshape(np.shape(gnom_range_xv), on_line=True, eps=1.0)

        # project to tangent image
        tangent_points = triangle_param["tangent_points"]
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2epr(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # get the tangent image pixels value
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list], padding_size, tangent_image_size)

        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y, tangent_image_x, channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255)

        # set the pixels outside the boundary to transparent
        tangent_image[:, :, 3] = 0
        tangent_image[tangent_image_y, tangent_image_x, 3] = 255
        tangent_image_list.append(tangent_image)

    return tangent_image_list


def ico2erp_image(tangent_images, erp_image_height, padding_size = 0.0):
    """Stitch the level-0 icosahedron's tangent image to ERP image.

    TODO there are seam on the stitched erp image.

    :param tangent_images: 20 tangent images in order.
    :type tangent_images: a list of numpy
    :param erp_image_height: the output erp image's height.
    :type erp_image_height: int
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
        triangle_param = get_icosahedron_parameters(triangle_index)

        # get all tangent triangle's available pixels
        availied_ERP_area = triangle_param["availied_ERP_area"]
        erp_image_row_start = int(availied_ERP_area[0] * (erp_image_height - 1))
        erp_image_row_stop = int(availied_ERP_area[1] * (erp_image_height - 1))
        erp_image_col_start = int(availied_ERP_area[2] * (erp_image_width - 1))
        erp_image_col_stop = int(availied_ERP_area[3] * (erp_image_width - 1))
        triangle_x_range = np.linspace(erp_image_col_start, erp_image_col_stop, erp_image_col_stop - erp_image_col_start + 1)
        triangle_y_range = np.linspace(erp_image_row_start, erp_image_row_stop, erp_image_row_stop - erp_image_row_start + 1)
        triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)

        # project spherical coordinate to tangent plane
        spherical_uv = sc.erp2sph([triangle_xv, triangle_yv], erp_image_height=erp_image_height)
        lambda_0 = triangle_param["tangent_points"][0]
        phi_1 = triangle_param["tangent_points"][1]
        tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], lambda_0, phi_1)

        # the pixels in the tangent triangle
        triangle_points_tangent = triangle_param["triangle_points_tangent"]
        available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                     triangle_points_tangent, on_line=True, eps=1e-3).reshape(tangent_xv.shape)

        # sample the pixel from the tangent image
        tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list], \
            padding_size, tangent_image_width, tangent_image_height)
        for channel in range(0, images_channels_number):
            erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] = \
                ndimage.map_coordinates(tangent_images[triangle_index][:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)

    return erp_image


def erp2ico_flow(erp_flow, tangent_image_size):
    """Project the ERP flow to the 20 tangent flows.

    :param erp_flow: The ERP flow image.
    :type erp_flow: numpy
    :param tangent_image_size: [description]
    :type tangent_image_size: [type]
    :return: a list ontain 20 triangle images
    :type list of numpy
    """
    pass


def ico2erp_flow(tangent_flows, erp_image_height):
    """Stitch all 20 tangent flows to a ERP flow.

    :param tangent_flows: The list of 20 tangnet flow data.
    :type tangent_flows: list of numpy 
    :param erp_image_height: the height of stitched ERP flow image 
    :type erp_image_height: int
    :return: the stitched ERP flow
    :type numpy
    """
    pass
