

import numpy as np
from scipy import ndimage

import image_io
import flow_io
import flow_vis
import flow_warp
import gnomonic_projection
import spherical_coordinates


"""
Convention of cubemap:
1) 6 face order is +x, -x, +y, -y, +z, -z; 
2) The gnomonic projection result (normalized tangent image) origin at the center, +x is right, + y is up. And the tangent point is the origin of image.

Reference: https://en.wikipedia.org/wiki/Cube_mapping
    In the spherical coordinate systehm the forward is +z, down is +y, right is +x.  The center of ERP's theta (latitude) and phi(longitude) is (0,0) 
"""


def generage_cubic_ply(mesh_file_path):
    """
    :param mesh_file_path: output ply file path
    """
    radius = 1

    cubemap_points = get_cubemap_parameters()

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


def get_cubemap_parameters():
    """
    Get the information of circumscribed cuboid in spherical coordinate system:
    0) tangent points;
    1) 4 corner points for each tangent images;

    The points order is: TL->TR->BR->BL

    :return: a dict the (phi, theta)
    """
    cubemap_point_theta = np.arctan(np.sqrt(2.0) * 0.5)

    # 1) get the tangent points (phi, theta)
    tangent_points_list = np.zeros((6, 2), dtype=float)
    tangent_points_list[0] = [np.pi / 2.0, 0]  # +x
    tangent_points_list[1] = [-np.pi / 2.0, 0]  # -x
    tangent_points_list[2] = [0.0, -np.pi / 2.0]  # +y
    tangent_points_list[3] = [0.0, np.pi / 2.0]  # -y
    tangent_points_list[4] = [0.0, 0.0]  # +z
    tangent_points_list[5] = [-np.pi, 0.0]  # -z

    # 2) 4 point (phi, theta) for 6 face's 4 3D points of circumscribed cuboid
    face_points_list = np.zeros((6, 4, 2), dtype=float)
    # Face 0, +x
    face_idx = 0
    face_points_list[face_idx][0] = [0.25 * np.pi, cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [0.75 * np.pi, cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [0.25 * np.pi, -cubemap_point_theta]  # BL

    # Face 1, -x
    face_idx = 1
    face_points_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [-0.25 * np.pi, cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [-0.25 * np.pi, -cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_theta]  # BL

    # Face 2, +y
    face_idx = 2
    face_points_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [0.25 * np.pi, cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [-0.25 * np.pi, cubemap_point_theta]  # BL

    # Face 3, -y
    face_idx = 3
    face_points_list[face_idx][0] = [-0.25 * np.pi, -cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [0.25 * np.pi, -cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_theta]  # BL

    # Face 4, +z
    face_idx = 4
    face_points_list[face_idx][0] = [-0.25 * np.pi, cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [0.25 * np.pi, cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [0.25 * np.pi, -cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [-0.25 * np.pi, -cubemap_point_theta]  # BL

    # Face 5, -z
    face_idx = 5
    face_points_list[face_idx][0] = [0.75 * np.pi, cubemap_point_theta]  # TL
    face_points_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_theta]  # TR
    face_points_list[face_idx][2] = [-0.75 * np.pi, -cubemap_point_theta]  # BR
    face_points_list[face_idx][3] = [0.75 * np.pi, -cubemap_point_theta]  # BL

    # 3) the cubemap face range in the ERP image
    face_erp_project_range = np.zeros((7, 4, 2), dtype=float)
    # Face 0, +x
    face_idx = 0
    face_erp_project_range[face_idx][0] = [0.25 * np.pi, 0.25 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [0.75 * np.pi, 0.25 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [0.75 * np.pi, -0.25 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [0.25 * np.pi, -0.25 * np.pi]  # BL

    # Face 1, -x
    face_idx = 1
    face_erp_project_range[face_idx][0] = [-0.75 * np.pi, 0.25 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [-0.25 * np.pi, 0.25 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [-0.25 * np.pi, -0.25 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [-0.75 * np.pi, -0.25 * np.pi]  # BL

    # Face 2, +y
    face_idx = 2
    face_erp_project_range[face_idx][0] = [-np.pi, -cubemap_point_theta]  # TL
    face_erp_project_range[face_idx][1] = [np.pi, -cubemap_point_theta]  # TR
    face_erp_project_range[face_idx][2] = [np.pi, -0.5 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [-np.pi, -0.5 * np.pi]  # BL

    # Face 3, -y
    face_idx = 3
    face_erp_project_range[face_idx][0] = [-np.pi, 0.5 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [np.pi, 0.5 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [np.pi, cubemap_point_theta]  # BR
    face_erp_project_range[face_idx][3] = [-np.pi, cubemap_point_theta]  # BL

    # Face 4, +z
    face_idx = 4
    face_erp_project_range[face_idx][0] = [-0.25 * np.pi, 0.25 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [0.25 * np.pi, 0.25 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [0.25 * np.pi, -0.25 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [-0.25 * np.pi, -0.25 * np.pi]  # BL

    # Face 5, -z, compose with two piece of image
    face_idx = 5
    # the first piece of image in the left of ERP image
    face_erp_project_range[face_idx][0] = [-np.pi, 0.25 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [-0.75 * np.pi, 0.25 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [-0.75 * np.pi, -0.25 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [-np.pi, -0.25 * np.pi]  # BL
    face_idx = 6
    # the second piece of image in the right of ERP image
    face_erp_project_range[face_idx][0] = [0.75 * np.pi, 0.25 * np.pi]  # TL
    face_erp_project_range[face_idx][1] = [np.pi, 0.25 * np.pi]  # TR
    face_erp_project_range[face_idx][2] = [np.pi, -0.25 * np.pi]  # BR
    face_erp_project_range[face_idx][3] = [0.75 * np.pi, -0.25 * np.pi]  # BL

    return {"tangent_points": tangent_points_list, "face_points": face_points_list, "face_erp_range": face_erp_project_range}


def erp2cubemap_flow(erp_flow_mat):
    """
    Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_flow: the equirectangular image's flow, dimension is [height, width, 3]
    :retrun: 6 images of each fact of cubemap projection
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_flows = []
    face_image_size = 500  # the size of each face
    erp_image_height = np.shape(erp_flow_mat)[0]
    erp_image_width = np.shape(erp_flow_mat)[1]
    erp_image_channel = np.shape(erp_flow_mat)[2]

    cubemap_points = get_cubemap_parameters()
    tangent_points_list = cubemap_points["tangent_points"]

    for index in range(0, 6):
        center_point = tangent_points_list[index]

        # 0) Get the location of tangent image's pixels corresponding location in ERP

        # tangent center project point
        lambda_0 = center_point[0]
        phi_1 = center_point[1]

        # the x,y of tangent image
        x_grid = np.linspace(-1.0, 1.0, face_image_size)
        y_grid = np.linspace(1.0, -1.0, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        lambda_, phi_ = gnomonic_projection.reverse_gnomonic_projection(x, y, lambda_0, phi_1)

        # spherical coordinate to pixel location
        erp_pixel_x = ((lambda_ + np.pi) / (2 * np.pi)) * erp_image_width
        erp_pixel_y = (- phi_ + 0.5 * np.pi) / np.pi * erp_image_height

        # process warp around, make the range in [0, image_width), and [0, image_height)
        erp_pixel_x[erp_pixel_x < 0] = erp_pixel_x[erp_pixel_x < 0] + erp_image_width
        erp_pixel_x[erp_pixel_x >= erp_image_width] = erp_pixel_x[erp_pixel_x >= erp_image_width] - erp_image_width
        erp_pixel_y[erp_pixel_y < 0] = erp_pixel_y[erp_pixel_y < 0] + erp_image_height
        erp_pixel_y[erp_pixel_y >= erp_image_height] = erp_pixel_y[erp_pixel_y >= erp_image_height] - erp_image_height

        # interpollation
        erp_pixel_flow = np.zeros((face_image_size, face_image_size, erp_image_channel), dtype=float)
        for channel in range(0, erp_image_channel):
            erp_pixel_flow[:, :, channel] = ndimage.map_coordinates(erp_flow_mat[:, :, channel], [erp_pixel_y, erp_pixel_x], order=1, mode='wrap')
            # erp_pixel_flow[:, :, channel] = ndimage.map_coordinates(erp_flow_mat[:, :, channel].T, [erp_pixel_x, erp_pixel_y], order=1)

        # 1) comput the end point location in the tangent image
        # convert the ERP optical flow's UV to tangent image's UV
        erp_pixel_x_target = erp_pixel_x + erp_pixel_flow[:, :, 0]
        erp_pixel_y_target = erp_pixel_y + erp_pixel_flow[:, :, 1]
        # process warp around
        erp_pixel_x_target[erp_pixel_x_target < 0] = erp_pixel_x_target[erp_pixel_x_target < 0] + erp_image_width
        erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] = erp_pixel_x_target[erp_pixel_x_target >= erp_image_width] - erp_image_width
        erp_pixel_y_target[erp_pixel_y_target < 0] = erp_pixel_y_target[erp_pixel_y_target < 0] + erp_image_height
        erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] = erp_pixel_y_target[erp_pixel_y_target >= erp_image_height] - erp_image_height
        # convert the erp location to spherical coordinate location
        lambda_target = erp_pixel_x_target / erp_image_width * np.pi * 2 - np.pi
        phi_target = -erp_pixel_y_target / erp_image_height * np.pi + 0.5 * np.pi
        # spherical location to tangent location
        face_image_x_target, face_image_y_target = gnomonic_projection.gnomonic_projection(lambda_target, phi_target, lambda_0, phi_1)
        face_flow_u = (face_image_x_target - x) * face_image_size * 0.5
        face_flow_v = (face_image_y_target - y) * face_image_size * 0.5
        face_flow_v = -face_flow_v  # transform to image coordinate system (+y is to down)

        # 2) the optical flow of tangent image
        face_flow = np.stack((face_flow_u, face_flow_v), axis=2)
        cubmap_tangent_flows.append(face_flow)

    return cubmap_tangent_flows


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

    cubemap_points = get_cubemap_parameters()
    tangent_points_list = cubemap_points["tangent_points"]

    for index in range(0, 6):
        center_point = tangent_points_list[index]

        # tangent center project point
        lambda_0 = center_point[0]
        phi_1 = center_point[1]

        # the xy of tangent image
        x_grid = np.linspace(-1.0, 1.0, face_image_size)
        y_grid = np.linspace(1.0, -1.0, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        lambda_, phi_ = gnomonic_projection.reverse_gnomonic_projection(x, y, lambda_0, phi_1)

        # spherical coordinate to pixel location
        erp_pixel_x = ((lambda_ + np.pi) / (2 * np.pi)) * erp_image_width
        erp_pixel_y = (-phi_ + np.pi / 2.0) / np.pi * erp_image_height

        # process warp around
        erp_pixel_x[erp_pixel_x < 0] = erp_pixel_x[erp_pixel_x < 0] + erp_image_width
        erp_pixel_x[erp_pixel_x >= erp_image_width] = erp_pixel_x[erp_pixel_x >= erp_image_width] - erp_image_width

        erp_pixel_y[erp_pixel_y < 0] = erp_pixel_y[erp_pixel_y < 0] + erp_image_height
        erp_pixel_y[erp_pixel_y >= erp_image_height] = erp_pixel_y[erp_pixel_y >= erp_image_height] - erp_image_height

        # interpollation
        face_image = np.zeros((face_image_size, face_image_size, erp_image_channel), dtype=float)
        for channel in range(0, erp_image_channel):
            face_image[:, :, channel] = ndimage.map_coordinates(erp_image_mat[:, :, channel], [erp_pixel_y, erp_pixel_x], order=1, mode='wrap')

        cubmap_tangent_images.append(face_image)

    return cubmap_tangent_images


def cubemap2erp_image(cubemap_images_list):
    """
    Assamble the 6 face cubemap to ERP image.

    :param cubemap_list: cubemap images, the sequence is +x, -x, +y, -y, +z, -z
    :return: The ERP RGB image the output image width is 4 times of cubmap's size
    """
    # check the face images number
    if not 6 == len(cubemap_images_list):
        raise RuntimeError("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_images_list[0])[0]
    # assert cubemap_image_size == 7
    erp_image_width = int(cubemap_image_size * 4.0)
    erp_image_height = int(erp_image_width * 0.5)
    erp_image_channel = np.shape(cubemap_images_list[0])[2]
    erp_image_mat = np.zeros((erp_image_height, erp_image_width, 3), dtype=np.float64)

    cubemap_points = get_cubemap_parameters()
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]  # 7 pieces of image

    for image_index in range(0, 7):
        # 0) tangent center project point
        if image_index == 6:
            face_index = image_index - 1
        else:
            face_index = image_index

        # get the tangent ERP image pixel's spherical coordinate location
        # the spherical coordinate range for each face
        face_phi_min = face_erp_range_sphere_list[image_index][3][0]
        face_theta_min = face_erp_range_sphere_list[image_index][3][1]
        face_erp_x_min, face_erp_y_max = spherical_coordinates.spherical2epr(face_phi_min, face_theta_min, erp_image_height)

        face_phi_max = face_erp_range_sphere_list[image_index][1][0]
        face_theta_max = face_erp_range_sphere_list[image_index][1][1]
        face_erp_x_max, face_erp_y_min = spherical_coordinates.spherical2epr(face_phi_max, face_theta_max, erp_image_height)

        # process the boundary of image
        face_erp_x_min = int(face_erp_x_min) if int(face_erp_x_min) > 0 else 0
        face_erp_x_max = int(face_erp_x_max + 0.5) if int(face_erp_x_max + 0.5) < erp_image_width else erp_image_width - 1
        face_erp_y_min = int(face_erp_y_min) if int(face_erp_y_min + 0.5) > 0 else 0
        face_erp_y_max = int(face_erp_y_max + 0.5) if int(face_erp_y_max + 0.5) < erp_image_height else erp_image_height - 1

        # 1) get ERP image's pixels corresponding tangent image location
        center_point = tangent_points_list[face_index]
        lambda_0 = center_point[0]
        phi_1 = center_point[1]
        face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
        face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
        face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
        face_phi_, face_theta_ = spherical_coordinates.erp2spherical((face_erp_x, face_erp_y), erp_image_height)

        # 2) get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        face_x, face_y = gnomonic_projection.gnomonic_projection(face_phi_, face_theta_, lambda_0, phi_1)
        pbc = 1.0  # projection_boundary_coefficient
        inside_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x.flatten(), face_y.flatten()), axis=1),
                                                            np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]]), True).reshape(np.shape(face_x))

        # remove the pixels outside the tangent image & translate to tangent image pixel coordinate,
        face_x_available = (face_x[inside_list] + 1.0) * 0.5 * (cubemap_image_size - 1)
        face_y_available = -(face_y[inside_list] - 1.0) * 0.5 * (cubemap_image_size - 1)

        # 3) get the value of interpollations
        for channel in range(0, erp_image_channel):
            erp_image_mat[face_erp_y[inside_list].astype(np.int64), face_erp_x[inside_list].astype(np.int64), channel] = \
                ndimage.map_coordinates(cubemap_images_list[face_index][:, :, channel], [face_y_available, face_x_available],
                                        order=1, mode='constant', cval=255)

    return erp_image_mat


def cubemap2erp_flow(cubemap_flows_list):
    """
    Assamble the 6 cubemap optical flow to ERP image. 

    :param cubemap_flows_list: the images sequence is +x, -x, +y, -y, +z, -z
    :return: the ERP flow image the image size 
    """
    # check the face images number
    if not 6 == len(cubemap_flows_list):
        raise RuntimeError("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_flows_list[0])[0]
    erp_flow_width = int(cubemap_image_size * 4.0)
    erp_flow_height = int(erp_flow_width * 0.5)
    erp_flow_channel = np.shape(cubemap_flows_list[0])[2]
    if not erp_flow_channel == 2:
        raise RuntimeError("The flow channels number is {}".format(erp_flow_channel))

    erp_flow_mat = np.zeros((erp_flow_height, erp_flow_width, 2), dtype=np.float64)

    cubemap_points = get_cubemap_parameters()
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]  # 7 pieces of image

    for flow_index in range(0, 7):
        # 0) tangent center project point
        if flow_index == 6:
            face_index = flow_index - 1
        else:
            face_index = flow_index

        # get the tangent ERP image pixel's spherical coordinate location range for each face
        face_phi_min = face_erp_range_sphere_list[flow_index][3][0]
        face_theta_min = face_erp_range_sphere_list[flow_index][3][1]
        face_erp_x_min, face_erp_y_max = spherical_coordinates.spherical2epr(face_phi_min, face_theta_min, erp_flow_height)

        face_phi_max = face_erp_range_sphere_list[flow_index][1][0]
        face_theta_max = face_erp_range_sphere_list[flow_index][1][1]
        face_erp_x_max, face_erp_y_min = spherical_coordinates.spherical2epr(face_phi_max, face_theta_max, erp_flow_height)

        center_point = tangent_points_list[face_index]
        lambda_0 = center_point[0]
        phi_1 = center_point[1]

        # process the boundary of image TODO there is seam in the ERP image!!
        face_erp_x_min = int(face_erp_x_min) if int(face_erp_x_min) > 0 else 0
        face_erp_x_max = int(face_erp_x_max + 0.5) if int(face_erp_x_max + 0.5) < erp_flow_width else erp_flow_width - 1
        face_erp_y_min = int(face_erp_y_min) if int(face_erp_y_min) > 0 else 0
        face_erp_y_max = int(face_erp_y_max + 0.5) if int(face_erp_y_max + 0.5) < erp_flow_height else erp_flow_height - 1

        # 2) get the pixels tangent image space location

        # ERP image space --> spherical space
        face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
        face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
        face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
        face_phi_, face_theta_ = spherical_coordinates.erp2spherical((face_erp_x, face_erp_y), erp_flow_height)

        # spherical space --> normailzed tangent image space
        face_x_src, face_y_src = gnomonic_projection.gnomonic_projection(face_phi_, face_theta_, lambda_0, phi_1)
        available_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x_src.flatten(), face_y_src.flatten()), axis=1),
                                                               np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])).reshape(np.shape(face_x_src))

        # normailzed tangent image space --> tangent image space
        face_x_src = (face_x_src + 1.0) * 0.5 * (cubemap_image_size - 1)
        face_y_src = -(face_y_src - 1.0) * 0.5 * (cubemap_image_size - 1)

        # 3) get the value of interpollations
        # 3-0) remove the pixels outside the tangent image
        face_x_src_available = face_x_src[available_list]
        face_y_src_available = face_y_src[available_list]
        # get the tangent images flow in the tangent image space
        face_flow_x = ndimage.map_coordinates(cubemap_flows_list[face_index][:, :, 0],
                                              [face_y_src_available, face_x_src_available],
                                              order=1, mode='constant', cval=255)
        face_x_tar_available = face_x_src_available + face_flow_x

        face_flow_y = ndimage.map_coordinates(cubemap_flows_list[face_index][:, :, 1],
                                              [face_y_src_available, face_x_src_available],
                                              order=1, mode='constant', cval=255)
        face_y_tar_available = face_y_src_available + face_flow_y

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        face_x_tar_available = face_x_tar_available / (0.5 * cubemap_image_size) - 1.0
        face_y_tar_available = -face_y_tar_available / (0.5 * cubemap_image_size) + 1.0

        # tangent normailzed space --> spherical space
        face_phi_tar, face_theta_tar = gnomonic_projection.reverse_gnomonic_projection(face_x_tar_available, face_y_tar_available, lambda_0, phi_1)

        # spherical space --> ERP image space
        face_x_tar_available, face_y_tar_available = spherical_coordinates.spherical2epr(face_phi_tar, face_theta_tar, erp_flow_height)

        # 4) get ERP flow with source and target pixels location
        face_flow_u = face_x_tar_available - face_erp_x[available_list]
        face_flow_v = face_y_tar_available - face_erp_y[available_list]

        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 0] = face_flow_u
        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 0] = face_flow_v

    return erp_flow_mat
