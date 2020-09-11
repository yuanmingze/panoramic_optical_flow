import math

import numpy as np
from utility import image_io
from utility import nfov

def generate_icosphere():
    """
    """
    mesh_file_path = "icosphere.ply"

    r = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0,   r, 0.0],
        [ 1.0,   r, 0.0],
        [-1.0,  -r, 0.0],
        [ 1.0,  -r, 0.0],
        [0.0, -1.0,   r],
        [0.0,  1.0,   r],
        [0.0, -1.0,  -r],
        [0.0,  1.0,  -r],
        [  r, 0.0, -1.0],
        [  r, 0.0,  1.0],
        [ -r, 0.0, -1.0],
        [ -r, 0.0,  1.0],
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
    with open(mesh_file_path,'w') as mesh_file:
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
    pass

def sphere2tangent():
    """
    project the equirectangular image to tangent triangle image.
    """
    image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    tangent_image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb_00.jpg"
    image_data = image_io.image_read(image_path)

    image_height = np.shape(image_data)[0]
    image_width = np.shape(image_data)[1]

    # compute the tangent image size
    tangent_image_width =  int(image_width / 5)
    tangent_image_hfov = 1. / 5.0
    tangent_image_height = int(image_height * ((0.5 * np.pi - np.arctan(0.5)) / np.pi))
    tangent_image_vfov = (0.5 * np.pi - np.arctan(0.5)) / np.pi

    # compute the tangent image with 3D point 
    # # 0) 
    # triangle_col_number_list = np.linspace(1, tangent_image_width, tangent_image_height)
    # icosphere_vertice_00 = np.array([0, 0 , 1])
    # icosphere_vertice_01 = np.array()
    # # for row_

    # import ipdb; ipdb.set_trace()
    # compute the tangent image with gnomonic projection
    # gnomonic_proj_center_sph_theta = np.pi * 2 / 5.0 / 2.0

    # transforme the erp image to tangent image
    theta_1 = 0.0

    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * ( 3 + np.sqrt(5))
    phi_0 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed)

    nfov_obj = nfov.NFOV(height=tangent_image_height, width=tangent_image_width)
    nfov_obj.FOV = [tangent_image_vfov, tangent_image_hfov] # 
    # the theta_1 and phi_0

    # the center point location [0,1] in ERP image, top-left is [0,0]
    # gnomonic_proj_center_sph
    center_point = np.array([(theta_1 - np.pi) / (2 * np.pi), phi_0/ np.pi], np.float)  # camera center point (valid range [0,1])
    tangent_image = nfov_obj.toNFOV(image_data, center_point)
    image_io.image_save(tangent_image, tangent_image_path)


def tangent2sphere():
    """
    transfrom the tangent image back to erp image.
    """
    image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    tangent_image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb_00.jpg"

    pass

if __name__ == "__main__":
    # sphere2tangent()
    # generate_icosphere()
    sphere2tangent()