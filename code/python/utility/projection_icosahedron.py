import math
import numpy as np


from . import image_io
from . import nfov
from . import gnomonic_projection as gp 

"""
Implement the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""

def get_tangent_point(triangle_index):
    """
    Return the tangent point theta and phi. Known as the lambda_0 and phi_1.
    """
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    lambda_0 = -1
    phi_1 = -1

    lambda_step = 2.0 * np.pi / 5.0

    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        phi_1 = phi_1_up

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index = triangle_index - 5
        phi_1_middle_up = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1 = phi_1_middle_up

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index = triangle_index - 10
        phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = phi_1_middle_down

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index = triangle_index - 15
        phi_1_down = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = phi_1_down

    return lambda_0, phi_1

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



def tangent2sphere(tangent_image_folder, output_folder, erp_image_size):
    """
    project a tangent images back to ERP image.
    :param erp_image_path: the input tangent image folder
    """
    tangent_image_file_name = r"erp_{:04d}.png"
    erp_image_height = erp_image_size[0]
    erp_image_width = erp_image_size[1]

    tangent_image_name = tangent_image_file_name.format(0)
    tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

    tangent_image_height = np.shape(tangent_image_data)[0]
    tangent_image_width = np.shape(tangent_image_data)[1]

    # 0) the up 5 triangles
    print("0) the up 5 triangles")
    for triangle_index in range(0, 5):
        step_number = triangle_index - 0

        erp_image = np.full(erp_image_size, 255)

        tangent_image_name = tangent_image_file_name.format(triangle_index)
        tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

        #
        tangent_triangle_points = get_tangent_triangle_points(triangle_index)

        index_x_start = 0
        index_x_stop = 480 - 1

        index_y_start = 0
        index_y_stop = 480 - 1

        lambda_0, phi_1 = get_tangent_point(triangle_index)

        for index_y in range(index_y_start, index_y_stop):
            for index_x in range(index_x_start, index_x_stop):

                tangent_x = (index_x - tangent_image_width / 2.0) / (tangent_image_width / 2.0)
                tangent_y = (index_y - tangent_image_height / 2.0) / (tangent_image_height / 2.0)

                lambda_, phi = reverse_gnomonic_projection(tangent_x, tangent_y, lambda_0, phi_1)

                erp_index_x = int((lambda_ + np.pi) / (2 * np.pi) * erp_image_width)
                erp_index_y = int((-phi + np.pi / 2.0) / np.pi * erp_image_height)

                #print(erp_index_x, erp_index_y, index_x, index_y)
                erp_image[erp_index_y, erp_index_x] = tangent_image_data[index_y, index_x, 0:3]

    # 1) the middle 10 triangles
    print("1) the middle 10 triangles")
    # 1-0) the middle-up 5 triangles
    print("1-0) the middle-up 5 triangles")

    # 1-1) the middle-down 5 triangles
    print("1-1) the middle-down 5 triangles")

    # 2) the down 5 triangles
    print("2) the down 5 triangles")


def get_tangent_point(triangle_index):
    """
    Return the tangent point theta and phi. Known as the lambda_0 and phi_1.
    """
    # def get_tangent_triangle_3D_points(triangle_index):
    #     """
    #     Return the specified triangle 3D points in Cartesian coordinate system.
    #     :param triangle_index: 
    #     """
    #     pass


    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    lambda_0 = -1
    phi_1 = -1

    lambda_step = 2.0 * np.pi / 5.0

    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        phi_1 = phi_1_up

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index = triangle_index - 5
        phi_1_middle_up = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        lambda_0 = - np.pi + lambda_step / 2.0 + triangle_index * lambda_step
        phi_1 = phi_1_middle_up

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index = triangle_index - 10
        phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = phi_1_middle_down

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index = triangle_index - 15
        phi_1_down = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        lambda_0 = - np.pi + triangle_index * lambda_step
        phi_1 = phi_1_down

    return lambda_0, phi_1


def get_tangent_triangle_points(triangle_index):
    """
    Return the specified triangle 3 points' spherical coordinate.

    :param triangle_index:
    """
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # 1) the up 5 triangles
    if triangle_index >= 0 and triangle_index <= 4:
        phi_up_center_step = 2.0 * np.pi / 5.0
        lambda_0 = - np.pi + phi_up_center_step / 2.0 + triangle_index * phi_up_center_step
        phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * phi_up_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.pi / 2.0, lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + triangle_index * phi_up_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + (triangle_index + 1) * phi_up_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_02))

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    phi_middle_center_step = 2.0 * np.pi / 5.0

    if triangle_index >= 5 and triangle_index <= 9:
        triangle_index = triangle_index - 5
        lambda_0 = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        phi_1_middle_up = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + (triangle_index + 1) * phi_middle_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_02))

    # 2-1) the middle-down triangles
    if triangle_index >= 10 and triangle_index <= 14:
        triangle_index = triangle_index - 10
        lambda_0 = - np.pi + triangle_index * phi_middle_center_step
        phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = - np.pi - phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        # cross the ERP image boundary
        if triangle_index == 10:
            triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_02))

    # 3) the down 5 triangles
    if triangle_index >= 15 and triangle_index <= 19:
        triangle_index = triangle_index - 15
        phi_down_center_step = 2.0 * np.pi / 5.0
        lambda_0 = - np.pi + triangle_index * phi_up_center_step
        phi_1_down = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = - np.pi - phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = - np.pi + phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        # cross the ERP image boundary
        if triangle_index == 15:
            triangle_point_01_lambda = triangle_point_01_lambda + 2 * np.pi
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = - np.pi + triangle_index * phi_down_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, -np.pi / 2.0, lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_02))


def ico2erp_image_gnomonic(tangent_image_folder, output_folder, erp_image_size):
    """
    project a tangent images back to ERP image.

    :param erp_image_path: the input tangent image folder
    """
    tangent_image_file_name = r"erp_{:04d}.png"
    erp_image_height = erp_image_size[0]
    erp_image_width = erp_image_size[1]

    tangent_image_name = tangent_image_file_name.format(0)
    tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

    tangent_image_height = np.shape(tangent_image_data)[0]
    tangent_image_width = np.shape(tangent_image_data)[1]

    # 0) the up 5 triangles
    print("0) the up 5 triangles")
    for triangle_index in range(0, 5):
        step_number = triangle_index - 0

        erp_image = np.full(erp_image_size, 255)

        tangent_image_name = tangent_image_file_name.format(triangle_index)
        tangent_image_data = image_io.image_read(tangent_image_folder + tangent_image_name)

        #
        tangent_triangle_points = get_tangent_triangle_points(triangle_index)

        index_x_start = 0
        index_x_stop = 480 - 1

        index_y_start = 0
        index_y_stop = 480 - 1

        lambda_0, phi_1 = get_tangent_point(triangle_index)

        for index_y in range(index_y_start, index_y_stop):
            for index_x in range(index_x_start, index_x_stop):

                tangent_x = (index_x - tangent_image_width / 2.0) / (tangent_image_width / 2.0)
                tangent_y = (index_y - tangent_image_height / 2.0) / (tangent_image_height / 2.0)

                lambda_, phi = reverse_gnomonic_projection(tangent_x, tangent_y, lambda_0, phi_1)

                erp_index_x = int((lambda_ + np.pi) / (2 * np.pi) * erp_image_width)
                erp_index_y = int((-phi + np.pi / 2.0) / np.pi * erp_image_height)

                #print(erp_index_x, erp_index_y, index_x, index_y)
                erp_image[erp_index_y, erp_index_x] = tangent_image_data[index_y, index_x, 0:3]

    # 1) the middle 10 triangles
    print("1) the middle 10 triangles")
    # 1-0) the middle-up 5 triangles
    print("1-0) the middle-up 5 triangles")

    # 1-1) the middle-down 5 triangles
    print("1-1) the middle-down 5 triangles")

    # 2) the down 5 triangles
    print("2) the down 5 triangles")


def erp2ico_image_gnomonic(erp_image_path, output_folder):
    """
    Generate tangent images with icosahedron projection. 
    With Gnomonic projection.

    TODO: process the pixels in the boundary of triangles
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

    # tangent image size for one hemisphere
    tangent_image_size = 480

    # 1) the up 5 triangles
    print("1) the up 5 triangles")
    phi_up_center_step = 2.0 * np.pi / 5.0
    phi_1_up = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

    for triangle_index in range(0, 5):
        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + phi_up_center_step / 2.0 + triangle_index * phi_up_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * phi_up_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.pi / 2.0, lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + triangle_index * phi_up_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + (triangle_index + 1) * phi_up_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, np.arctan(0.5), lambda_0, phi_1_up)
        triangle_points.append(np.array(triangle_point_02))

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
                if not inside_triangle(triangle_points, np.array([x, y])):
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
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = -np.pi + (triangle_index + 1) * phi_middle_center_step
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = -np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_up)
        triangle_points.append(np.array(triangle_point_02))

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
                if not inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(int(triangle_index + 5)))

    # 2-1) the middle-down triangles
    print("2-1) the middle-down triangles")
    phi_1_middle_down = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

    for triangle_index in range(11, 15):
        triangle_index = triangle_index - 10

        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + triangle_index * phi_middle_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = -np.pi + triangle_index * phi_middle_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = - np.pi - phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        # TODO For cross boundary
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = - np.pi + phi_middle_center_step / 2.0 + triangle_index * phi_middle_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, - np.arctan(0.5), lambda_0, phi_1_middle_down)
        triangle_points.append(np.array(triangle_point_02))

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
                if not inside_triangle(triangle_points, np.array([x, y])):
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

    for triangle_index in range(16, 20):
        triangle_index = triangle_index - 15

        tangent_image = np.full([tangent_image_size, tangent_image_size, 3], 255)

        lambda_0 = - np.pi + triangle_index * phi_down_center_step
        # the tangent triangle points coordinate in tangent image
        triangle_points = []
        triangle_point_00_lambda = - np.pi - phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        triangle_point_00 = gp.gnomonic_projection(triangle_point_00_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_00))

        triangle_point_01_lambda = - np.pi + phi_down_center_step / 2.0 + triangle_index * phi_down_center_step
        # TODO For cross boundary
        triangle_point_01 = gp.gnomonic_projection(triangle_point_01_lambda, -np.arctan(0.5), lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_01))

        triangle_point_02_lambda = - np.pi + triangle_index * phi_down_center_step
        triangle_point_02 = gp.gnomonic_projection(triangle_point_02_lambda, -np.pi / 2.0, lambda_0, phi_1_down)
        triangle_points.append(np.array(triangle_point_02))

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
                if not inside_triangle(triangle_points, np.array([x, y])):
                    continue

                tangent_image_y = - y * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image_x = x * tangent_image_size / 2.0 + tangent_image_size / 2.0
                tangent_image[int(tangent_image_y), int(tangent_image_x)] = image_data[index_y, index_x]

        # image_io.image_show(tangent_image)
        image_io.image_save_rgba(tangent_image.astype(np.uint8), tangent_image_root + tangent_image_file_name.format(triangle_index + 15))


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


def erp2ico_nfov():
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


def ico2erp_nfov():
    """
    projection the tangent image to sphere 
    """
    pass