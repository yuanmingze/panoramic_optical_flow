

import math
import numpy as np
from scipy import ndimage

from . import image_io
from . import flow_io
from . import flow_vis

from . import tangent_image

"""
Convention of cubemap:
1) 6 face order is +x, -x, +y, -y, +z, -z; 
2) 

Reference: https://en.wikipedia.org/wiki/Cube_mapping
    In the spherical coordinate systehm the forward is +z, down is +y, right is +x.  The center of ERP's theta (latitude) and phi(longitude) is (0,0) 
"""


def generage_cubic_ply(mesh_file_path):
    """
    :param mesh_file_path: output ply file path
    """
    radius = 1

    cubemap_points = get_cubemap_points()

    # add 4 corner points
    face_points = cubemap_points["face_points"].reshape((-1, 2))

    # add tangent points
    tangent_points = cubemap_points["tangent_points"]
    face_points = np.concatenate((face_points, tangent_points), axis=0)

    # to 3D space
    phi = face_points[:, 0]
    theta = face_points[:, 1]

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.cos(theta) * np.cos(phi)
    z = radius * np.sin(theta)

    vertices = np.stack((x, y, z), 1)

    # faces = np.array([
    #     [0, 11, 5],
    #     [0, 5, 1],
    #     [0, 1, 7],
    #     [0, 7, 10],
    #     [0, 10, 11],
    #     [1, 5, 9],
    #     [5, 11, 4],
    #     [11, 10, 2],
    #     [10, 7, 6],
    #     [7, 1, 8],
    #     [3, 9, 4],
    #     [3, 4, 2],
    #     [3, 2, 6],
    #     [3, 6, 8],
    #     [3, 8, 9],
    #     [5, 4, 9],
    #     [2, 4, 11],
    #     [6, 2, 10],
    #     [8, 6, 7],
    #     [9, 8, 1],
    # ])

    # output the to obj file
    with open(mesh_file_path, 'w') as mesh_file:
        # output header
        mesh_file.write("ply\n")
        mesh_file.write("format ascii 1.0\n")
        mesh_file.write("element vertex {}\n".format(np.shape(vertices)[0]))
        mesh_file.write("property float x\n")
        mesh_file.write("property float y\n")
        mesh_file.write("property float z\n")
        # mesh_file.write("element face {}\n".format(np.shape(faces)[0]))
        # mesh_file.write("property list uchar int vertex_index\n")
        mesh_file.write("end_header\n")
        for index in range(np.shape(vertices)[0]):
            mesh_file.write("{} {} {}\n".format(vertices[index][0], vertices[index][1], vertices[index][2]))

        # for index in range(np.shape(faces)[0]):
        #     mesh_file.write("3 {} {} {}\n".format(faces[index][0], faces[index][1], faces[index][2]))


def get_cubemap_points():
    """
    Get the information of circumscribed cuboid in spherical coordinate system:
    0) tangent points;
    1) 4 corner points for each tangent images;

    The points order is: TL->TR->BR->BL

    :return: a dict the (phi, theta)
    """
    # get the tangent points (phi, theta)
    tangent_points_list = np.zeros((6, 2), dtype=float)
    tangent_points_list[0] = [np.pi / 2.0, 0]  # +x
    tangent_points_list[1] = [-np.pi / 2.0, 0]  # -x
    tangent_points_list[2] = [0.0, -np.pi / 2.0]  # +y
    tangent_points_list[3] = [0.0, np.pi / 2.0]  # -y
    tangent_points_list[4] = [0.0, 0.0]  # +z
    tangent_points_list[5] = [-np.pi, 0.0]  # -z

    # 4 point (phi, theta) for 6 face's 4 3D points of circumscribed cuboid
    face_points_list = np.zeros((6, 4, 2), dtype=float)
    # Face 0, +x
    face_idx = 0
    face_points_list[face_idx][0] = [1.0 / 4.0 * np.pi, np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [3.0 / 4.0 * np.pi, np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [1.0 / 4.0 * np.pi, -np.pi / 4.0]  # BL

    # Face 1, -x
    face_idx = 1
    face_points_list[face_idx][0] = [-3.0 / 4.0 * np.pi, np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [-1.0 / 4.0 * np.pi, np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [-1.0 / 4.0 * np.pi, -np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [-3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BL

    # Face 2, +y
    face_idx = 2
    face_points_list[face_idx][0] = [-3.0 / 4.0 * np.pi, np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [-3.0 / 4.0 * np.pi, np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [1.0 / 4.0 * np.pi, np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [-1.0 / 4.0 * np.pi, np.pi / 4.0]  # BL

    # Face 3, -y
    face_idx = 3
    face_points_list[face_idx][0] = [-1.0 / 4.0 * np.pi, -np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [1.0 / 4.0 * np.pi, -np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [-3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BL

    # Face 4, +z
    face_idx = 4
    face_points_list[face_idx][0] = [-1.0 / 4.0 * np.pi, np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [1.0 / 4.0 * np.pi, np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [1.0 / 4.0 * np.pi, -np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [-1.0 / 4.0 * np.pi, -np.pi / 4.0]  # BL

    # Face 5, -z
    face_idx = 5
    face_points_list[face_idx][0] = [3.0 / 4.0 * np.pi, np.pi / 4.0]  # TL
    face_points_list[face_idx][1] = [-3.0 / 4.0 * np.pi, np.pi / 4.0]  # TR
    face_points_list[face_idx][2] = [-3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BR
    face_points_list[face_idx][3] = [3.0 / 4.0 * np.pi, -np.pi / 4.0]  # BL

    return {"tangent_points": tangent_points_list, "face_points": face_points_list}


def erp2cubemap_flow(erp_flow_mat):
    """
    Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_flow: the equirectangular image's flow, dimension is [height, width, 3]
    :retrun: 6 images of each fact of cubemap projection
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_images = []
    face_image_size = 500  # the size of each face
    erp_image_height = np.shape(erp_flow_mat)[0]
    erp_image_width = np.shape(erp_flow_mat)[1]
    erp_image_channel = np.shape(erp_flow_mat)[2]

    cubemap_points = get_cubemap_points()
    tangent_points_list = cubemap_points["tangent_points"]
    face_points_list = cubemap_points["face_points"]

    for index in range(0, 6):
        center_point = tangent_points_list[index]
        face_point = face_points_list[index]

        # 0) Get the location of tangent image's pixels corresponding location in ERP

        # tangent center project point
        lambda_0 = center_point[0]
        phi_1 = center_point[1]

        # the x,y of tangent image
        x_grid = np.linspace(-1.0, 1.0, face_image_size)
        y_grid = np.linspace(1.0, -1.0, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        lambda_, phi_ = tangent_image.reverse_gnomonic_projection(x, y, lambda_0, phi_1)

        # spherical coordinate to pixel location
        erp_pixel_x = ((lambda_ + np.pi) / (2 * np.pi)) * erp_image_width
        erp_pixel_y = (- phi_ + np.pi / 2.0) / np.pi * erp_image_height

        # process warp around, make the range in [0, image_width), and [0, image_height)
        erp_pixel_x[erp_pixel_x < 0] = erp_pixel_x[erp_pixel_x < 0] + erp_image_width
        erp_pixel_x[erp_pixel_x >= erp_image_width] = erp_pixel_x[erp_pixel_x >= erp_image_width] - erp_image_width
        erp_pixel_y[erp_pixel_y < 0] = erp_pixel_y[erp_pixel_y < 0] + erp_image_height
        erp_pixel_y[erp_pixel_y >= erp_image_height] = erp_pixel_y[erp_pixel_y >= erp_image_height] - erp_image_height

        # interpollation
        face_image = np.zeros((face_image_size, face_image_size, erp_image_channel), dtype=float)
        for channel in range(0, erp_image_channel):
            face_image[:, :, channel] = ndimage.map_coordinates(erp_flow_mat[:, :, channel], [erp_pixel_y, erp_pixel_x], order=1, mode = 'wrap')

        # 1) comput the end point location in the tangent image

        # convert the ERP optical flow's UV to tangent image's UV
        erp_pixel_x_target = erp_pixel_x + face_image[:,:,0]
        erp_pixel_y_target = erp_pixel_y + face_image[:,:,1]
        # process warp around
        erp_pixel_x_target[erp_pixel_x_target < 0] = erp_pixel_x_target[erp_pixel_x_target < 0] + erp_image_width
        erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] = erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] - erp_image_width
        erp_pixel_y_target[erp_pixel_y_target < 0] = erp_pixel_y_target[erp_pixel_y_target < 0] + erp_image_height
        erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] = erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] - erp_image_height
        # convert the erp location to spherical coordinate location
        lambda_target = erp_pixel_x_target / erp_image_width * np.pi * 2 - np.pi
        phi_target = -erp_pixel_y_target / erp_image_height * np.pi + np.pi / 2.0
        # spherical location to tangent location
        face_image_x_target, face_image_y_target = tangent_image.gnomonic_projection(lambda_target, phi_target, lambda_0, phi_1)
        face_flow_u = (face_image_x_target - x) * face_image_size
        face_flow_v = (face_image_y_target - y) * face_image_size

        # 2) the optical flow of tangent image
        face_image_flow = np.stack((face_flow_u, face_flow_v))

        # image_io.image_show(face_image)
        # image_io.image_show(phi_)
        # image_io.image_show(lambda_)
        # image_io.image_show(erp_pixel_x)
        # image_io.image_show(erp_pixel_y)
        cubmap_tangent_images.append(face_image)

    return cubmap_tangent_images


def erp2cubemap_image(erp_image_mat):
    """
    Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_flow: the equirectangular image's flow, dimension is [height, width, 3]
    :retrun: 6 images of each fact of cubemap projection
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_images = []
    face_image_size = 500  # the size of each face
    erp_image_height = np.shape(erp_image_mat)[0]
    erp_image_width = np.shape(erp_image_mat)[1]
    erp_image_channel = np.shape(erp_image_mat)[2]

    cubemap_points = get_cubemap_points()
    tangent_points_list = cubemap_points["tangent_points"]
    face_points_list = cubemap_points["face_points"]

    for index in range(0, 6):
        center_point = tangent_points_list[index]
        face_point = face_points_list[index]

        # tangent center project point
        lambda_0 = center_point[0]
        phi_1 = center_point[1]

        # the xy of tangent image
        x_grid = np.linspace(-1.0, 1.0, face_image_size)
        y_grid = np.linspace(1.0, -1.0, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        lambda_, phi_ = tangent_image.reverse_gnomonic_projection(x, y, lambda_0, phi_1)

        # spherical coordinate to pixel location
        erp_pixel_x = ((lambda_ + np.pi) / (2 * np.pi)) * erp_image_width
        erp_pixel_y = (- phi_ + np.pi / 2.0) / np.pi * erp_image_height

        # process warp around
        erp_pixel_x[erp_pixel_x < 0] = erp_pixel_x[erp_pixel_x < 0] + erp_image_width
        erp_pixel_x[erp_pixel_x >= erp_image_width] = erp_pixel_x[erp_pixel_x >= erp_image_width] - erp_image_width

        erp_pixel_y[erp_pixel_y < 0] = erp_pixel_y[erp_pixel_y < 0] + erp_image_height
        erp_pixel_y[erp_pixel_y >= erp_image_height] = erp_pixel_y[erp_pixel_y >= erp_image_height] - erp_image_height

        # interpollation
        # from scipy.interpolate import griddata
        # erp_grid_x, erp_grid_y = np.mgrid[0:1:(erp_image_width * 1j), 0:1:(erp_image_height * 1j)]

        # grid_z0 = griddata(points, values, (erp_grid_x, erp_grid_y), method='linear')
        face_image = np.zeros((face_image_size, face_image_size, erp_image_channel), dtype=float)
        for channel in range(0, erp_image_channel):
            face_image[:, :, channel] = ndimage.map_coordinates(erp_image_mat[:, :, channel], [erp_pixel_y, erp_pixel_x], order=1, mode = 'wrap')
        # image_io.image_show(face_image)
        # image_io.image_show(phi_)
        # image_io.image_show(lambda_)
        # image_io.image_show(erp_pixel_x)
        # image_io.image_show(erp_pixel_y)
        cubmap_tangent_images.append(face_image)

    return cubmap_tangent_images


def cubemap2erp_image(cubemap_images_list):
    """
    Assamble the 6 face cubemap to ERP image.

    :param cubemap_list: the cubemap 
    :return: The ERP RGB image
    """
    pass


def cubemap2erp_flow(cubemap_flows_list):
    """
    Assamble the 6 cubemap optical flow to ERP image. 

    :param cubemap_flows_list:
    :return: the ERP flow image
    """
    pass


if __name__ == "__main__":
    # Test
    # # 1) output cubemap ply mesh
    # cubemap_ply_filepath = "../../data/cubemap_points.ply"
    # generage_cubic_ply(cubemap_ply_filepath)

    # # 2) erp image to cube map
    # erp_image_filepath = "../../data/replica_360/hotel_0/0001_rgb.jpg"
    # cubemap_images_output = "../../data/"
    # erp_image = image_io.image_read(erp_image_filepath)
    # face_images = erp2cubemap_image(erp_image)
    # for index in range(0, len(face_images)):
    #     cubemap_images_name = cubemap_images_output + "cubemap_rgb_{}.jpg".format(index)
    #     image_io.image_save(face_images[index], cubemap_images_name)
    #     # image_io.image_show(face_images[0])

    # 3) erp flow to cube map
    # erp_flow_filepath = "../../data/replica_360/hotel_0/0001_opticalflow_forward.flo" 
    erp_flow_filepath = "/mnt/sda1/workdata/opticalflow_data/replica_360/apartment_0/replica_seq_data/0001_opticalflow_backward.flo"
    cubemap_flow_output = "../../data/"
    erp_flow = flow_io.readFlowFile(erp_flow_filepath)
    face_flows = erp2cubemap_flow(erp_flow)
    for index in range(0, len(face_flows)):
        cubemap_flow_name = cubemap_flow_output + "cubemap_flo_{}.jpg".format(index)
        face_flow_vis = flow_vis.flow_to_color(face_flows[index])
        image_io.image_save(face_flow_vis, cubemap_flow_name)
        # image_io.image_show(face_flow_vis)