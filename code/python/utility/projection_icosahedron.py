import math
import copy
import numpy as np
from scipy import ndimage


from . import image_io
from . import nfov
from . import gnomonic_projection as gp
from . import spherical_coordinates


from .logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Implement the Gnomonic projection (forward and reverse projection).
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


# def tangent2sphere(tangent_image_folder, output_folder, erp_image_size):
#     """
#     project a tangent images back to ERP image.
#     :param erp_image_path: the input tangent image folder
#     """
#     tangent_image_file_name = r"erp_{:04d}.png"
#     erp_image_height = erp_image_size[0]
#     erp_image_width = erp_image_size[1]

#     tangent_image_name = tangent_image_file_name.format(0)
#     tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

#     tangent_image_height = np.shape(tangent_image_data)[0]
#     tangent_image_width = np.shape(tangent_image_data)[1]

#     # 0) the up 5 triangles
#     print("0) the up 5 triangles")
#     for triangle_index in range(0, 5):
#         step_number = triangle_index - 0

#         erp_image = np.full(erp_image_size, 255)

#         tangent_image_name = tangent_image_file_name.format(triangle_index)
#         tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

#         #
#         tangent_triangle_points = get_tangent_triangle_points(triangle_index)

#         index_x_start = 0
#         index_x_stop = 480 - 1

#         index_y_start = 0
#         index_y_stop = 480 - 1

#         lambda_0, phi_1 = get_tangent_point(triangle_index)

#         for index_y in range(index_y_start, index_y_stop):
#             for index_x in range(index_x_start, index_x_stop):

#                 tangent_x = (index_x - tangent_image_width / 2.0) / (tangent_image_width / 2.0)
#                 tangent_y = (index_y - tangent_image_height / 2.0) / (tangent_image_height / 2.0)

#                 lambda_, phi = reverse_gnomonic_projection(tangent_x, tangent_y, lambda_0, phi_1)

#                 erp_index_x = int((lambda_ + np.pi) / (2 * np.pi) * erp_image_width)
#                 erp_index_y = int((-phi + np.pi / 2.0) / np.pi * erp_image_height)

#                 #print(erp_index_x, erp_index_y, index_x, index_y)
#                 erp_image[erp_index_y, erp_index_x] = tangent_image_data[index_y, index_x, 0:3]

#     # 1) the middle 10 triangles
#     print("1) the middle 10 triangles")
#     # 1-0) the middle-up 5 triangles
#     print("1-0) the middle-up 5 triangles")

#     # 1-1) the middle-down 5 triangles
#     print("1-1) the middle-down 5 triangles")

#     # 2) the down 5 triangles
#     print("2) the down 5 triangles")


# def get_icosahedron_parameters():
#     """
#     Get icosahedron's tangent face's paramters.

#     :return the each tangent face's tangent point and 3 vertices in spherical coordinate system.
#     """
#     tangent_points_list = []
#     face_points_list = []
#     face_erp_project_range = []

#     for i in range(0, 20):
#         tangent_points_list.append(get_tangent_point(i))
#         face_points_list.append(get_tangent_triangle_points(i))


def get_icosahedron_parameters(triangle_index):
    """
    Get icosahedron's tangent face's paramters.
    Get the tangent point theta and phi. Known as the lambda_0 and phi_1.

    :return the tangent face's tangent point and 3 vertices's location.
    """

    radius_circumscribed = np.sin(2 * np.pi / 5.0)  # the
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    lambda_0 = None
    phi_1 = None
    triangle_point_00_lambda = None
    triangle_point_00_phi = None
    triangle_point_01_lambda = None
    triangle_point_01_phi = None
    triangle_point_02_lambda = None
    triangle_point_02_phi = None

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

    tangent_point = [lambda_0, phi_1]

    triangle_points_sph = []
    triangle_points_sph.append([triangle_point_00_lambda, triangle_point_00_phi])
    triangle_points_sph.append([triangle_point_01_lambda, triangle_point_01_phi])
    triangle_points_sph.append([triangle_point_02_lambda, triangle_point_02_phi])

    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_lambda, triangle_point_00_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_lambda, triangle_point_01_phi, lambda_0, phi_1))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_lambda, triangle_point_02_phi, lambda_0, phi_1))

    return {"tangent_points": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph}


# def get_tangent_triangle_points(triangle_index):
#     """
#     Get the specified triangle's 3 points in spherical coordinate.

#     :param triangle_index the tangent face index
#     :retrun the face's 3 vertices in spherical coordinate
#     """
#     triangle_points_tangent = []
#     triangle_point_00_tangent = None
#     triangle_point_01_tangent = None
#     triangle_point_02_tangent = None

#     triangle_points_sph = []
#     triangle_point_00_sph = None
#     triangle_point_01_sph = None
#     triangle_point_02_sph = None

#     # 1) the up 5 triangles
#     if triangle_index >= 0 and triangle_index <= 4:
#         phi_up_center_step = 2.0 * np.pi / 5.0
#         lambda_0 = - np.pi + phi_up_center_step / 2.0 + triangle_index * phi_up_center_step
#         phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
#         # the tangent triangle points coordinate in tangent image
#         triangle_point_00_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * phi_up_center_step
#         triangle_point_00_tangent = gp.gnomonic_projection(triangle_point_00_lambda, np.pi / 2.0, lambda_0, phi_1_up)

#         triangle_point_01_lambda = -np.pi + triangle_index * phi_up_center_step
#         triangle_point_01_tangent = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_up)

#         triangle_point_02_lambda = -np.pi + (triangle_index + 1) * phi_up_center_step
#         triangle_point_02_tangent = gp.gnomonic_projection(triangle_point_02_lambda, np.arctan(0.5), lambda_0, phi_1_up)

#     # 2) the middle 10 triangles
#     # 2-0) middle-up triangles
#     phi_middle_center_step = 2.0 * np.pi / 5.0

#     if triangle_index >= 5 and triangle_index <= 9:
#         triangle_index = triangle_index - 5
#         lambda_0 = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
#         phi_1_middle_up = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
#         # the tangent triangle points coordinate in tangent image
#         triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
#         triangle_point_00_sph = [triangle_point_00_lambda, np.arctan(0.5)]
#         triangle_point_00_tangent = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)

#         triangle_point_01_lambda = -np.pi + (triangle_index + 1) * phi_middle_center_step
#         triangle_point_01_tangent = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)

#         triangle_point_02_lambda = -np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
#         triangle_point_02_tangent = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_up)

#     # 2-1) the middle-down triangles
#     if triangle_index >= 10 and triangle_index <= 14:
#         triangle_index = triangle_index - 10
#         lambda_0 = - np.pi + triangle_index * phi_middle_center_step
#         phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
#         # the tangent triangle points coordinate in tangent image
#         triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
#         triangle_point_00_tangent = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_down)

#         triangle_point_01_lambda = - np.pi - phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
#         # cross the ERP image boundary
#         if triangle_index == 10:
#             triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
#         triangle_point_01_tangent = gp.gnomonic_projection(triangle_point_01_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)

#         triangle_point_02_lambda = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
#         triangle_point_02_tangent = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)

#     # 3) the down 5 triangles
#     if triangle_index >= 15 and triangle_index <= 19:
#         triangle_index = triangle_index - 15
#         phi_down_center_step = 2.0 * np.pi / 5.0
#         lambda_0 = - np.pi + triangle_index * phi_down_center_step
#         phi_1_down = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

#         # the tangent triangle points coordinate in tangent image
#         triangle_point_00_lambda = - np.pi - phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
#         triangle_point_00_tangent = gp.gnomonic_projection(triangle_point_00_lambda, -np.arctan(0.5), lambda_0, phi_1_down)

#         triangle_point_01_lambda = - np.pi + phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
#         # cross the ERP image boundary
#         if triangle_index == 15:
#             triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
#         triangle_point_01_tangent = gp.gnomonic_projection(triangle_point_01_lambda, -np.arctan(0.5), lambda_0, phi_1_down)

#         triangle_point_02_lambda = - np.pi + triangle_index * phi_down_center_step
#         triangle_point_02_tangent = gp.gnomonic_projection(triangle_point_02_lambda, -np.pi / 2.0, lambda_0, phi_1_down)

#     triangle_points_tangent.append([triangle_point_00_tangent])
#     triangle_points_tangent.append([triangle_point_01_tangent])
#     triangle_points_tangent.append([triangle_point_02_tangent])
#     return triangle_points


# def ico2erp_image_gnomonic(tangent_image_folder, output_folder, erp_image_size):
#     """
#     project a tangent images back to ERP image.

#     :param erp_image_path: the input tangent image folder
#     """
#     tangent_image_file_name = r"erp_{:04d}.png"
#     erp_image_height = erp_image_size[0]
#     erp_image_width = erp_image_size[1]

#     tangent_image_name = tangent_image_file_name.format(0)
#     tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

#     tangent_image_height = np.shape(tangent_image_data)[0]
#     tangent_image_width = np.shape(tangent_image_data)[1]

#     # 0) the up 5 triangles
#     print("0) the up 5 triangles")
#     for triangle_index in range(0, 5):
#         step_number = triangle_index - 0

#         erp_image = np.full(erp_image_size, 255)

#         tangent_image_name = tangent_image_file_name.format(triangle_index)
#         tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

#         #
#         tangent_triangle_points = get_tangent_triangle_points(triangle_index)

#         index_x_start = 0
#         index_x_stop = 480 - 1

#         index_y_start = 0
#         index_y_stop = 480 - 1

#         lambda_0, phi_1 = get_tangent_point(triangle_index)

#         for index_y in range(index_y_start, index_y_stop):
#             for index_x in range(index_x_start, index_x_stop):

#                 tangent_x = (index_x - tangent_image_width / 2.0) / (tangent_image_width / 2.0)
#                 tangent_y = (index_y - tangent_image_height / 2.0) / (tangent_image_height / 2.0)

#                 lambda_, phi = reverse_gnomonic_projection(tangent_x, tangent_y, lambda_0, phi_1)

#                 erp_index_x = int((lambda_ + np.pi) / (2 * np.pi) * erp_image_width)
#                 erp_index_y = int((-phi + np.pi / 2.0) / np.pi * erp_image_height)

#                 #print(erp_index_x, erp_index_y, index_x, index_y)
#                 erp_image[erp_index_y, erp_index_x] = tangent_image_data[index_y, index_x, 0:3]

#     # 1) the middle 10 triangles
#     print("1) the middle 10 triangles")
#     # 1-0) the middle-up 5 triangles
#     print("1-0) the middle-up 5 triangles")

#     # 1-1) the middle-down 5 triangles
#     print("1-1) the middle-down 5 triangles")

#     # 2) the down 5 triangles
#     print("2) the down 5 triangles")


def erp2ico_image_gnomonic(erp_image, tangent_image_size):
    """Project the equirectangular image to 20 triangle images.

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
    gnom_range_x = np.linspace(-1, 1, tangent_image_size)
    gnom_range_y = np.linspace(-1, 1, tangent_image_size)
    gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

    # generate tangent images
    for triangle_index in range(0, 20):
        log.debug("generate the tangent image {}".format(triangle_index))
        tangent_image = np.full([tangent_image_size, tangent_image_size, 4], 255)
        triangle_param = get_icosahedron_parameters(triangle_index)

        # the tangent triangle points coordinate in tangent image
        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        inside_list = gp.inside_polygon_2d(np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1), tangent_triangle_vertices).reshape(np.shape(gnom_range_xv))

        # project to tangent image
        tangent_points = triangle_param["tangent_points"]
        tangent_triangle_lambda_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_points[0], tangent_points[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = spherical_coordinates.spherical2epr(tangent_triangle_lambda_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # get the tangent image pixels value
        tangent_image_x = gnom_range_xv[inside_list] * tangent_image_size * 0.5 + tangent_image_size * 0.5
        tangent_image_y = -gnom_range_yv[inside_list] * tangent_image_size * 0.5 + tangent_image_size * 0.5
        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y.astype(np.int), tangent_image_x.astype(np.int), channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='constant', cval=255)

        # set the pixels outside the boundary to transparent
        tangent_image[:,:, 3] = 0
        tangent_image[tangent_image_y.astype(np.int), tangent_image_x.astype(np.int), 3] = 255
        tangent_image_list.append(tangent_image)

    return tangent_image_list


def erp2ico_image_gnomonic_backup(erp_image_path, output_folder, tangent_image_size=480):
    """Project the equirectangular image to 20 triangle images.

    TODO: process the pixels in the boundary of triangles

    :param erp_image: the input equirectangular image.
    :type erp_image_path: numpy array, [height, width, 3]
    :param tangent_image_size: the output triangle image size, defaults to 480
    :type tangent_image_size: int, optional
    :return: a list contain 20 triangle images
    :type list
    """
    image_path = erp_image_path
    tangent_image_root = output_folder
    tangent_image_file_name = r"tangent_{:04d}.png"

    # ERP image size
    image_data = image_io.image_read(image_path)
    if np.shape(image_data)[2] == 4:
        image_data = image_data[:, :, 0:3]
    image_height = np.shape(image_data)[0]
    image_width = np.shape(image_data)[1]

    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # 1) the up 5 triangles
    print("1) the up 5 triangles")
    phi_up_center_step = 2.0 * np.pi / 5.0
    phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

    for triangle_index in range(0, 5):
        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + phi_up_center_step / 2.0 + triangle_index * phi_up_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_points_sph = []
        triangle_point_00_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * phi_up_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.pi / 2.0, lambda_0, phi_1_up)
        triangle_points_sph.append([triangle_point_00_lambda, np.pi / 2.0])
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + triangle_index * phi_up_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points_sph.append([triangle_point_01_lambda, np.arctan(0.5)])
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + (triangle_index + 1) * phi_up_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points_sph.append([triangle_point_02_lambda, np.arctan(0.5)])
        triangle_points.append(np.array(triangle_point_02))

        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(triangle_index, triangle_points, triangle_points_sph))

        # spherical coordinate (0,0) is in the center of ERP image
        erp_image_row_start = 0
        erp_image_row_stop = int((np.pi / 2 - np.arctan(0.5)) / np.pi * image_height)

        erp_image_col_start = math.floor(image_width / 5.0 * triangle_index)
        erp_image_col_stop = math.floor(image_width / 5.0 * (triangle_index + 1))

        for index_y in range(erp_image_row_start, erp_image_row_stop):
            for index_x in range(erp_image_col_start, erp_image_col_stop):
                lambda_ = index_x / image_width * 2 * np.pi - np.pi
                phi = (image_height / 2.0 - index_y) / (image_height / 2.0) * (np.pi / 2.0)

                # project to tangent image
                x, y = gp.gnomonic_projection(lambda_, phi, lambda_0, phi_1_up)

                # in the tangent image triangle?
                if not gp.inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(triangle_index))

    # 2) the middle 10 triangles
    print("2) the middle 10 triangles")
    # 2-0) middle-up triangles
    print("2-0) middle-up triangles")
    phi_middle_center_step = 2.0 * np.pi / 5.0
    phi_1_middle_up = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)

    for triangle_index in range(5, 10):
        triangle_index = (triangle_index - 5)

        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_points_sph = []
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points_sph.append([triangle_point_00_lambda, np.arctan(0.5)])
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + (triangle_index + 1) * phi_middle_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points_sph.append([triangle_point_01_lambda, np.arctan(0.5)])
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points_sph.append([triangle_point_02_lambda, -np.arctan(0.5)])
        triangle_points.append(np.array(triangle_point_02))

        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(triangle_index, triangle_points, triangle_points_sph))
        # spherical coordinate (0,0) is in the center of ERP image
        erp_image_row_start = int((np.arccos(radius_inscribed / radius_circumscribed) + np.arccos(radius_inscribed / radius_midradius)) / np.pi * image_height)
        erp_image_row_stop = int((np.pi / 2.0 + np.arctan(0.5)) / np.pi * image_height)

        erp_image_col_start = math.floor(image_width / 5.0 * triangle_index)
        erp_image_col_stop = math.floor(image_width / 5.0 * (triangle_index + 1))

        for index_y in range(erp_image_row_start, erp_image_row_stop):
            for index_x in range(erp_image_col_start, erp_image_col_stop):
                lambda_ = index_x / image_width * 2 * np.pi - np.pi
                phi = (image_height / 2.0 - index_y) / (image_height / 2.0) * (np.pi / 2.0)

                # project to tangent image
                x, y = gp.gnomonic_projection(lambda_, phi, lambda_0, phi_1_middle_up)

                # in the tangent image triangle?
                if not gp.inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(int(triangle_index + 5)))

    # 2-1) the middle-down triangles
    print("2-1) the middle-down triangles")
    phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

    for triangle_index in range(10, 15):
        triangle_index = triangle_index - 10

        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + triangle_index * phi_middle_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_points_sph = []
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_00))
        triangle_points_sph.append([triangle_point_00_lambda, np.arctan(0.5)])

        triangle_point_01_lambda = - np.pi - phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        # TODO For cross boundary
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_01))
        triangle_points_sph.append([triangle_point_01_lambda, -np.arctan(0.5)])

        triangle_point_02_lambda = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_02))
        triangle_points_sph.append([triangle_point_02_lambda, -np.arctan(0.5)])

        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(triangle_index, triangle_points, triangle_points_sph))
        # spherical coordinate (0,0) is in the center of ERP image
        erp_image_row_start = int((np.pi / 2.0 - np.arctan(0.5)) / np.pi * image_height)
        erp_image_row_stop = int((np.pi - np.arccos(radius_inscribed / radius_circumscribed) - np.arccos(radius_inscribed / radius_midradius)) / np.pi * image_height)

        erp_image_col_start = math.floor(image_width / 5.0 * triangle_index - image_width / 5.0 / 2.0)
        erp_image_col_stop = math.floor(image_width / 5.0 * triangle_index + image_width / 5.0 / 2.0)

        for index_y in range(erp_image_row_start, erp_image_row_stop):
            for index_x in range(erp_image_col_start, erp_image_col_stop):
                lambda_ = index_x / image_width * 2 * np.pi - np.pi
                phi = (image_height / 2.0 - index_y) / (image_height / 2.0) * (np.pi / 2.0)

                # project to tangent image
                x, y = gp.gnomonic_projection(lambda_, phi, lambda_0, phi_1_middle_down)

                # in the tangent image triangle?
                if not gp.inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(triangle_index + 10))

    # 3) the down 5 triangles
    print("3) the down 5 triangles")
    phi_down_center_step = 2.0 * np.pi / 5.0
    phi_1_down = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

    for triangle_index in range(15, 20):
        triangle_index = triangle_index - 15

        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + triangle_index * phi_down_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_points_sph = []
        triangle_point_00_lambda = - np.pi - phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_00))
        triangle_points_sph.append([triangle_point_00_lambda, -np.arctan(0.5)])

        triangle_point_01_lambda = - np.pi + phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        # TODO For cross boundary
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_01))
        triangle_points_sph.append([triangle_point_01_lambda, -np.arctan(0.5)])

        triangle_point_02_lambda = - np.pi + triangle_index * phi_down_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, -np.pi / 2.0, lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_02))
        triangle_points_sph.append([triangle_point_02_lambda,  -np.pi / 2.0])

        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(triangle_index, triangle_points, triangle_points_sph))
        # spherical coordinate (0,0) is in the center of ERP image
        erp_image_row_start = int((np.pi / 2.0 + np.arctan(0.5)) / np.pi * image_height)
        erp_image_row_stop = int(image_height)

        erp_image_col_start = math.floor(image_width / 5.0 * triangle_index - image_width / 5.0 / 2.0)
        erp_image_col_stop = math.floor(image_width / 5.0 * triangle_index + image_width / 5.0 / 2.0)

        for index_y in range(erp_image_row_start, erp_image_row_stop):
            for index_x in range(erp_image_col_start, erp_image_col_stop):
                lambda_ = index_x / image_width * 2 * np.pi - np.pi
                phi = (image_height / 2.0 - index_y) / (image_height / 2.0) * (np.pi / 2.0)

                # project to tangent image
                x, y = gp.gnomonic_projection(lambda_, phi, lambda_0, phi_1_down)

                # in the tangent image triangle?
                if not gp.inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(triangle_index + 15))


def ico2erp_image_nfov():
    """
    projection the tangent image to sphere 
    """
    pass


def erp2ico_image_nfov():
    """
    project the equirectangular image to tangent triangle image.
    """
    image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    tangent_image_root = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/"
    tangent_image_name = r"0001_rgb_{}_{:04d}.jpg"

    image_data = image_io.image_read(image_path)
    image_height = np.shape(image_data)[0]
    image_width = np.shape(image_data)[1]

    # compute the tangent image size
    tangent_image_width = int(image_width / 5)
    tangent_image_height_up = int(image_height * ((0.5 * np.pi - np.arctan(0.5)) / np.pi))
    tangent_image_height_middle = int(image_height * (2 * np.arctan(0.5)) / np.pi)
    tangent_image_height_down = int(image_height * ((0.5 * np.pi - np.arctan(0.5)) / np.pi))

    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)
    # transforme the erp image to tangent image to get the  icosahedron's 20 face

    # 1) the up 5
    tangent_image_hfov_up = 1. / 5.0
    tangent_image_vfov_up = (0.5 * np.pi - np.arctan(0.5)) / np.pi
    phi_up_center_step = 2.0 * np.pi / 5.0
    theta_up_center = np.arccos(radius_inscribed / radius_circumscribed)

    for face_index in range(0, 5):
        nfov_obj = nfov.NFOV(height=tangent_image_height_up, width=tangent_image_width)
        nfov_obj.FOV = [tangent_image_vfov_up, tangent_image_hfov_up]

        phi_center = face_index * phi_up_center_step + phi_up_center_step * 0.5
        theta_center = theta_up_center

        # nfov center is at top-left
        center_point = np.array([phi_center / (2 * np.pi), theta_center / np.pi], np.float)
        tangent_image = nfov_obj.toNFOV(image_data, center_point)

        tangent_image_path = tangent_image_root + tangent_image_name.format("up", face_index)
        image_io.image_save(tangent_image, tangent_image_path)

    # the middle 10 faces
    tangent_image_hfov_middle = 1. / 5.0
    tangent_image_vfov_middle = (2 * np.arctan(0.5)) / np.pi
    phi_middle_center_step = 2.0 * np.pi / 10.0
    theta_middle_center = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed)

    for face_index in range(0, 10):
        nfov_obj = nfov.NFOV(height=tangent_image_height_middle, width=tangent_image_width)
        nfov_obj.FOV = [tangent_image_vfov_middle, tangent_image_hfov_middle]

        phi_center = face_index * phi_middle_center_step  # + phi_middle_center_step * 0.5
        theta_center = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        if face_index % 2 == 1:
            theta_center = np.pi / 2.0 - theta_center
        else:
            theta_center = np.pi / 2.0 + theta_center

        # nfov center is at top-left
        center_point = np.array([phi_center / (2 * np.pi), theta_center / np.pi], np.float)
        tangent_image = nfov_obj.toNFOV(image_data, center_point)

        tangent_image_path = tangent_image_root + tangent_image_name.format("middle", face_index)
        image_io.image_save(tangent_image, tangent_image_path)

    # the down 5
    tangent_image_vfov_down = (0.5 * np.pi - np.arctan(0.5)) / np.pi
    tangent_image_hfov_down = 1. / 5.0
    phi_down_center_step = 2.0 * np.pi / 5.0
    theta_down_center = np.pi - np.arccos(radius_inscribed / radius_circumscribed)

    for face_index in range(0, 5):
        nfov_obj = nfov.NFOV(height=tangent_image_height_down, width=tangent_image_width)
        nfov_obj.FOV = [tangent_image_vfov_down, tangent_image_hfov_down]

        phi_center = face_index * phi_down_center_step
        theta_center = theta_down_center

        # nfov center is at top-left
        center_point = np.array([phi_center / (2 * np.pi), theta_center / np.pi], np.float)
        tangent_image = nfov_obj.toNFOV(image_data, center_point)

        tangent_image_path = tangent_image_root + tangent_image_name.format("down", face_index)
        image_io.image_save(tangent_image, tangent_image_path)


def erp2ico_image_3D():
    """
    Generate the tangent image with 3D tangent image plane.
    Return the specified triangle 3D points in Cartesian coordinate system.
    :param triangle_index: 
    """
    pass


def ico2erp_image_3D():
    """
    projection the tangent image to sphere 
    """
    pass
