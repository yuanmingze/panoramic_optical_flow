import math

import numpy as np
from utility import image_io
from utility import nfov
from utility import icosahedron

def generate_icosphere():
    """
    """
    mesh_file_path = "icosphere.ply"

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


def sphere2tanget3D():
    """
    """
    # compute the tangent image with 3D point
    # # 0)
    # triangle_col_number_list = np.linspace(1, tangent_image_width, tangent_image_height)
    # icosphere_vertice_00 = np.array([0, 0 , 1])
    # icosphere_vertice_01 = np.array()
    # # for row_

    # import ipdb; ipdb.set_trace()
    # compute the tangent image with gnomonic projection
    # gnomonic_proj_center_sph_theta = np.pi * 2 / 5.0 / 2.0
    pass

def sphere2tangent():
    """
    Generate tangent images with icosahedron projection.
    """
    pass

def sphere2tangent_nfov():
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

        phi_center = face_index * phi_middle_center_step #+ phi_middle_center_step * 0.5
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


def tangent2sphere_nfov():
    """
    projection the tangent image to sphere 
    """
    pass


def tangent2sphere():
    """
    transfrom the tangent image back to erp image.
    """
    image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    tangent_image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb_00.jpg"


if __name__ == "__main__":
    # sphere2tangent()
    # generate_icosphere()
    sphere2tangent_nfov()
