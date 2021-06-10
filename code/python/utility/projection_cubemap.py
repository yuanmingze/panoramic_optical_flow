import numpy as np
from scipy import ndimage

import gnomonic_projection
import polygon
import spherical_coordinates
import projection

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Cubemap for rgb image and optical flow:
1) 6 face order is +x, -x, +y, -y, +z, -z;   
Reference: https://en.wikipedia.org/wiki/Cube_mapping
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
    theta = face_points[:, 0]
    phi = face_points[:, 1]
    vertices = spherical_coordinates.sph2car(theta, phi, 1.0)

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
        # TODO add line
        # for index in range(np.shape(faces)[0]):
        #     mesh_file.write("3 {} {} {}\n".format(faces[index][0], faces[index][1], faces[index][2]))


def get_cubemap_parameters(padding_size=0.0):
    """
    Get the information of circumscribed cuboid in spherical coordinate system:
    0) tangent points;
    1) 4 corner points for each tangent images;
    2) tangent area range in spherical coordinate. And the points order is: TL->TR->BR->BL.

    :param padding_size: the padding size is base on the tangent image scale. 
    :type: float
    :return: the faces parameters
    :rtype: dict
    """
    cubemap_point_phi = np.arctan(np.sqrt(2.0) * 0.5)  # the poler of the point

    # 1) get the tangent points (theta, phi)
    tangent_center_points_list = np.zeros((6, 2), dtype=float)
    tangent_center_points_list[0] = [np.pi / 2.0, 0.0]  # +x
    tangent_center_points_list[1] = [-np.pi / 2.0, 0.0]  # -x
    tangent_center_points_list[2] = [0.0, -np.pi / 2.0]  # +y
    tangent_center_points_list[3] = [0.0, np.pi / 2.0]  # -y
    tangent_center_points_list[4] = [0.0, 0.0]  # +z
    tangent_center_points_list[5] = [-np.pi, 0.0]  # -z

    # 2) circumscribed cuboidfor 6 face's 4 3D point (theta, phi), unite sphere
    face_points_sph_list = np.zeros((6, 4, 2), dtype=float)
    # Face 0, +x
    face_idx = 0
    face_points_sph_list[face_idx][0] = [0.25 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [0.25 * np.pi, -cubemap_point_phi]  # BL

    # Face 1, -x
    face_idx = 1
    face_points_sph_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.25 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [-0.25 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_phi]  # BL

    # Face 2, +y
    face_idx = 2
    face_points_sph_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.25 * np.pi, cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.25 * np.pi, cubemap_point_phi]  # BL

    # Face 3, -y
    face_idx = 3
    face_points_sph_list[face_idx][0] = [-0.25 * np.pi, -cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.25 * np.pi, -cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_phi]  # BL

    # Face 4, +z
    face_idx = 4
    face_points_sph_list[face_idx][0] = [-0.25 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.25 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.25 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.25 * np.pi, -cubemap_point_phi]  # BL

    # Face 5, -z
    face_idx = 5
    face_points_sph_list[face_idx][0] = [0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [-0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [0.75 * np.pi, -cubemap_point_phi]  # BL

    # 3) the cubemap face range in the ERP image, the tangent range order is -x, +x , -y, +y
    face_erp_range = np.zeros((6, 4, 2), dtype=float)
    tangent_padded_range = 1.0 + padding_size
    for index in range(0, len(tangent_center_points_list)):
        tangent_center_point = tangent_center_points_list[index]
        theta_0 = tangent_center_point[0]
        phi_0 = tangent_center_point[1]
        if index == 0 or index == 1 or index == 4 or index == 5:
            # +x, -x , +z, -z
            # theta range
            theta_min, _ = gnomonic_projection.reverse_gnomonic_projection(-tangent_padded_range, 0, theta_0, phi_0)
            theta_max, _ = gnomonic_projection.reverse_gnomonic_projection(tangent_padded_range, 0, theta_0, phi_0)
            # phi (latitude) range
            _, phi_max = gnomonic_projection.reverse_gnomonic_projection(0, tangent_padded_range, theta_0, phi_0)
            _, phi_min = gnomonic_projection.reverse_gnomonic_projection(0, -tangent_padded_range, theta_0, phi_0)
        elif index == 2 or index == 3:
            # +y, -y
            _, phi_ = gnomonic_projection.reverse_gnomonic_projection(-tangent_padded_range, tangent_padded_range, theta_0, phi_0)
            if index == 2:  # +y
                theta_min = -np.pi
                theta_max = np.pi
                phi_max = phi_
                phi_min = -np.pi/2
            if index == 3:  # -y
                theta_min = -np.pi
                theta_max = np.pi
                phi_max = np.pi/2
                phi_min = phi_

        # the each face range in spherical coordinate
        face_erp_range[index][0] = [theta_min, phi_max]  # TL
        face_erp_range[index][1] = [theta_max, phi_max]  # TR
        face_erp_range[index][2] = [theta_max, phi_min]  # BR
        face_erp_range[index][3] = [theta_min, phi_min]  # BL

    return {"tangent_points": tangent_center_points_list,
            "face_points": face_points_sph_list,
            "face_erp_range": face_erp_range}


def erp2cubemap_flow(erp_flow_mat, padding_size=0.0, face_image_size=500):
    """
    Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_flow_mat: the equirectangular image's flow, dimension is [height, width, 3]
    :type erp_flow_mat: numpy
    :param padding_size: face flow padding size, defaults to 0.0
    :type padding_size: float, optional
    :param padding_size: the size of each face, defaults to 500
    :type padding_size: int, optional
    :retrun: 6 images of each fact of cubemap projection
    :rtype: list
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_flows = []
    erp_image_height = np.shape(erp_flow_mat)[0]
    erp_image_width = np.shape(erp_flow_mat)[1]
    erp_image_channel = np.shape(erp_flow_mat)[2]

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    gnomonic2image_ratio = (face_image_size - 1) / (2.0 + padding_size * 2.0)
    pbc = 1.0 + padding_size  # projection_boundary_coefficient

    for index in range(0, 6):
        center_point = tangent_points_list[index]

        # 0) Get the location of tangent image's pixels corresponding location in ERP
        # tangent center project point
        theta_0 = center_point[0]
        phi_0 = center_point[1]

        # the x,y of tangent image
        x_grid = np.linspace(-pbc, pbc, face_image_size)
        y_grid = np.linspace(pbc, - pbc, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        theta_, phi_ = gnomonic_projection.reverse_gnomonic_projection(x, y, theta_0, phi_0)

        # spherical coordinate to pixel location
        erp_pixel_x = ((theta_ + np.pi) / (2 * np.pi)) * erp_image_width
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
        theta_target = erp_pixel_x_target / erp_image_width * np.pi * 2 - np.pi
        phi_target = -erp_pixel_y_target / erp_image_height * np.pi + 0.5 * np.pi
        # spherical location to tangent location
        face_image_x_target, face_image_y_target = gnomonic_projection.gnomonic_projection(theta_target, phi_target, theta_0, phi_0)
        face_flow_u = (face_image_x_target - x) * gnomonic2image_ratio
        face_flow_v = (face_image_y_target - y) * gnomonic2image_ratio
        face_flow_v = -face_flow_v  # transform to image coordinate system (+y is to down)

        # 2) the optical flow of tangent image
        face_flow = np.stack((face_flow_u, face_flow_v), axis=2)
        cubmap_tangent_flows.append(face_flow)

    return cubmap_tangent_flows


def erp2cubemap_image(erp_image_mat, padding_size=0.0, face_image_size=None):
    """
    Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_image_mat: the equirectangular image, dimension is [height, width, 3]
    :type erp_image_mat: numpy 
    :param  face_image_size: the tangent face image size 
    :type face_image_size: int
    :param padding_size: the padding size outside the face boundary, defaults to 0.0, do not padding
    :type padding_size: the bound, optional
    :retrun: 6 images of each fact of cubemap projection
    :rtype: list
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_images = []

    erp_image_height = np.shape(erp_image_mat)[0]
    erp_image_width = np.shape(erp_image_mat)[1]
    erp_image_channel = np.shape(erp_image_mat)[2]

    if face_image_size is None:
        face_image_size = int(erp_image_width / 4.0)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    pbc = 1.0 + padding_size  # projection_boundary_coefficient

    for index in range(0, 6):
        center_point = tangent_points_list[index]

        # tangent center project point
        theta_0 = center_point[0]
        phi_0 = center_point[1]

        # the xy of tangent image
        x_grid = np.linspace(-pbc, pbc, face_image_size)
        y_grid = np.linspace(pbc, -pbc, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        theta_, phi_ = gnomonic_projection.reverse_gnomonic_projection(x, y, theta_0, phi_0)

        # spherical coordinate to pixel location
        erp_pixel_x = ((theta_ + np.pi) / (2 * np.pi)) * erp_image_width
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


def cubemap2erp_image(cubemap_images_list,  padding_size=0.0):
    """
    Assamble the 6 face cubemap to ERP image.

    :param cubemap_list: cubemap images, the sequence is +x, -x, +y, -y, +z, -z
    :return: The ERP RGB image the output image width is 4 times of cubmap's size
    """
    # check the face images number
    if not 6 == len(cubemap_images_list):
        log.error("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_images_list[0])[0]
    # assert cubemap_image_size == 7
    erp_image_width = int(cubemap_image_size * 4.0)
    erp_image_height = int(erp_image_width * 0.5)
    erp_image_channel = np.shape(cubemap_images_list[0])[2]
    erp_image_mat = np.zeros((erp_image_height, erp_image_width, 3), dtype=np.float64)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]  # 7 pieces of image

    for image_index in range(0, 6):
        # get the tangent ERP image pixel's spherical coordinate location
        # the spherical coordinate range for each face
        face_theta_min = face_erp_range_sphere_list[image_index][3][0]
        face_phi_min = face_erp_range_sphere_list[image_index][3][1]
        face_erp_x_min, face_erp_y_max = spherical_coordinates.sph2erp(face_theta_min, face_phi_min, erp_image_height, False)

        face_theta_max = face_erp_range_sphere_list[image_index][1][0]
        face_phi_max = face_erp_range_sphere_list[image_index][1][1]
        face_erp_x_max, face_erp_y_min = spherical_coordinates.sph2erp(face_theta_max, face_phi_max, erp_image_height, False)

        # process the image boundary
        face_erp_x_min = int(face_erp_x_min) if int(face_erp_x_min) > 0 else int(face_erp_x_min - 0.5)
        face_erp_x_max = int(face_erp_x_max + 0.5) if int(face_erp_x_max) > 0 else int(face_erp_x_max)
        face_erp_y_min = int(face_erp_y_min) if int(face_erp_y_min) > 0 else int(face_erp_y_min - 0.5)
        face_erp_y_max = int(face_erp_y_max + 0.5) if int(face_erp_y_max) > 0 else int(face_erp_y_max)

        # 1) get ERP image's pixels corresponding tangent image location
        face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
        face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
        face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
        face_erp_x = np.remainder(face_erp_x, erp_image_width)
        face_erp_y = np.remainder(face_erp_y, erp_image_height)
        face_theta_, face_phi_ = spherical_coordinates.erp2sph((face_erp_x, face_erp_y), erp_image_height, False)

        # 2) get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        center_point = tangent_points_list[image_index]
        theta_0 = center_point[0]
        phi_0 = center_point[1]
        face_x, face_y = gnomonic_projection.gnomonic_projection(face_theta_, face_phi_, theta_0, phi_0)
        pbc = 1.0 + padding_size  # projection_boundary_coefficient
        inside_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x.flatten(), face_y.flatten()), axis=1),
                                                            np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]]), True).reshape(np.shape(face_x))

        # remove the pixels outside the tangent image & translate to tangent image pixel coordinate,
        # map the gnomonic coordinate to tangent image's pixel coordinate.
        tangent_gnomonic_range = [-pbc, +pbc, -pbc, +pbc]
        face_x_available, face_y_available = gnomonic_projection.gnomonic2pixel(face_x[inside_list], face_y[inside_list],
                                                       0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # TODO blend boundary

        # 3) get the value of interpollations
        for channel in range(0, erp_image_channel):
            erp_image_mat[face_erp_y[inside_list].astype(np.int64), face_erp_x[inside_list].astype(np.int64), channel] = \
                ndimage.map_coordinates(cubemap_images_list[image_index][:, :, channel], [face_y_available, face_x_available],
                                        order=1, mode='constant', cval=255)

    return erp_image_mat


def cubemap2erp_flow(cubemap_flows_list, erp_flow_height=None, padding_size=0.0, image_erp_src=None, image_erp_tar=None, wrap_around = False):
    """
    Assamble the 6 cubemap optical flow to ERP optical flow. 

    :param cubemap_flows_list: the images sequence is +x, -x, +y, -y, +z, -z
    :type cubemap_flows_list: list
    :param erp_flow_height: the height of output flow 
    :type erp_flow_height: int
    :param padding_size: the cubemap's padding area size, defaults to 0.0
    :type padding_size: float, optional
    :param wrap_around: True, the optical flow is as perspective optical flow, False, it's warp around.
    :type wrap_around: bool
    :return: the ERP flow image the image size 
    :rtype: numpy
    """
    # check the face images number
    if not 6 == len(cubemap_flows_list):
        log.error("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_flows_list[0])[0]
    if erp_flow_height is None:
        erp_flow_height = cubemap_image_size * 2.0
    erp_flow_height = int(erp_flow_height)
    erp_flow_width = int(erp_flow_height * 2.0)
    erp_flow_channel = np.shape(cubemap_flows_list[0])[2]
    if not erp_flow_channel == 2:
        log.error("The flow channels number is {}".format(erp_flow_channel))

    erp_flow_mat = np.zeros((erp_flow_height, erp_flow_width, 2), dtype=np.float64)
    erp_flow_weight_mat = np.zeros((erp_flow_height, erp_flow_width), dtype=np.float64)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]
    pbc = 1.0 + padding_size  # projection_boundary_coefficient
    # gnomonic2image_ratio = (cubemap_image_size - 1) / (2.0 + padding_size * 2.0)

    for flow_index in range(0,  6):
        # get the tangent ERP image pixel's spherical coordinate location range for each face
        face_theta_min = np.amin(face_erp_range_sphere_list[flow_index][:, 0], axis=0)
        face_theta_max = np.amax(face_erp_range_sphere_list[flow_index][:, 0], axis=0)
        face_phi_min = np.amin(face_erp_range_sphere_list[flow_index][:, 1], axis=0)
        face_phi_max = np.amax(face_erp_range_sphere_list[flow_index][:, 1], axis=0)

        face_erp_x_min, face_erp_y_max = spherical_coordinates.sph2erp(face_theta_min, face_phi_min, erp_flow_height, False)
        face_erp_x_max, face_erp_y_min = spherical_coordinates.sph2erp(face_theta_max, face_phi_max, erp_flow_height, False)

        # process the boundary of image
        face_erp_x_min = int(face_erp_x_min) if int(face_erp_x_min) > 0 else int(face_erp_x_min - 0.5)
        face_erp_x_max = int(face_erp_x_max + 0.5) if int(face_erp_x_max) > 0 else int(face_erp_x_max)
        face_erp_y_min = int(face_erp_y_min) if int(face_erp_y_min) > 0 else int(face_erp_y_min - 0.5)
        face_erp_y_max = int(face_erp_y_max + 0.5) if int(face_erp_y_max) > 0 else int(face_erp_y_max)

        # 2) get the pixels location in tangent image location
        # ERP image space --> spherical space
        face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
        face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
        face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
        face_erp_x = np.remainder(face_erp_x, erp_flow_width)
        face_erp_y = np.remainder(face_erp_y, erp_flow_height)
        face_theta_, face_phi_ = spherical_coordinates.erp2sph((face_erp_x, face_erp_y), erp_flow_height, False)

        # spherical space --> normailzed tangent image space
        center_point = tangent_points_list[flow_index]
        theta_0 = center_point[0]
        phi_0 = center_point[1]
        face_x_src_gnomonic, face_y_src_gnomonic = gnomonic_projection.gnomonic_projection(face_theta_, face_phi_, theta_0, phi_0)

        available_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x_src_gnomonic.flatten(), face_y_src_gnomonic.flatten()), axis=1),
                                                               np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]]), True).reshape(np.shape(face_x_src_gnomonic))
        # normailzed tangent image space --> tangent image space
        tangent_gnomonic_range = [-pbc, +pbc, -pbc, +pbc]
        face_x_src_available, face_y_src_available = gnomonic_projection.gnomonic2pixel(face_x_src_gnomonic[available_list], face_y_src_gnomonic[available_list], 0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # 3) get the value of interpollations
        # 3-0) remove the pixels outside the tangent image
        # get ERP image's pixel available array, indicate pixels whether fall in the tangent face image
        # get the tangent images flow in the tangent image space
        face_flow_x = ndimage.map_coordinates(cubemap_flows_list[flow_index][:, :, 0], [face_y_src_available, face_x_src_available], order=1, mode='constant', cval=0.0)
        face_x_tar_pixel_available = face_x_src_available + face_flow_x
        face_flow_y = ndimage.map_coordinates(cubemap_flows_list[flow_index][:, :, 1], [face_y_src_available, face_x_src_available], order=1, mode='constant', cval=0.0)
        face_y_tar_pixel_available = face_y_src_available + face_flow_y

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        face_x_tar_gnomonic_available, face_y_tar_gnomonic_available = gnomonic_projection.pixel2gnomonic(face_x_tar_pixel_available, face_y_tar_pixel_available,0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # tangent normailzed space --> spherical space
        face_theta_tar, face_phi_tar = gnomonic_projection.reverse_gnomonic_projection(face_x_tar_gnomonic_available, face_y_tar_gnomonic_available, theta_0, phi_0)
        # spherical space --> ERP image space
        face_x_tar_available, face_y_tar_available = spherical_coordinates.sph2erp(face_theta_tar, face_phi_tar, erp_flow_height, True)

        # Process the face -z, -y, +y, cross the boundary
        if (flow_index == 5 or flow_index == 2 or flow_index == 3) and not wrap_around:
            log.info("ERP optical flow with wrap around. Face index {}".format(flow_index))
            face_x_src_gnomonic_available = face_x_src_gnomonic[available_list]
            face_y_src_gnomonic_available = face_y_src_gnomonic[available_list]
            p3 = np.stack((face_x_src_gnomonic_available, face_y_src_gnomonic_available))
            p4 = np.stack((face_x_tar_gnomonic_available, face_y_tar_gnomonic_available))
            # the line at tangent image cross the boundary (y axis of tangent image)
            gnomonic_max = 99999999
            if flow_index == 5:
                # cross_x_axis = polygon.detect_intersection_segments_array([0, -gnomonic_max], [0, +gnomonic_max], p3, p4)
                # cross_x_axis_plus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available > 0)
                # cross_x_axis_minus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available < 0)
                cross_x_axis_plus2pi = np.logical_and(face_x_src_gnomonic_available < 0, face_x_tar_gnomonic_available > 0)
                cross_x_axis_minus2pi = np.logical_and(face_x_src_gnomonic_available >= 0, face_x_tar_gnomonic_available < 0) 
            elif flow_index == 2:
                cross_x_axis = polygon.detect_intersection_segments_array([0, 0], [0, -gnomonic_max], p3, p4)
                cross_x_axis_plus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available < 0)
                cross_x_axis_minus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available > 0)
            elif flow_index == 3:
                cross_x_axis = polygon.detect_intersection_segments_array([0, 0], [0, gnomonic_max], p3, p4)
                cross_x_axis_plus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available < 0)
                cross_x_axis_minus2pi = np.logical_and(cross_x_axis, face_x_tar_gnomonic_available > 0)
            face_x_tar_available[cross_x_axis_minus2pi] = face_x_tar_available[cross_x_axis_minus2pi] - erp_flow_width
            face_x_tar_available[cross_x_axis_plus2pi] = face_x_tar_available[cross_x_axis_plus2pi] + erp_flow_width

        # 4) get ERP flow with source and target pixels location
        # 4-0) the ERP flow
        face_flow_u = face_x_tar_available - face_erp_x[available_list]
        face_flow_v = face_y_tar_available - face_erp_y[available_list]

        # 4-1) blend the optical flow
        # comput the all available pixels' weight
        if image_erp_src is not None and image_erp_tar is not None:
            weight_type = "normal_distribution_flowcenter"
            face_weight_mat_1 = projection.get_blend_weight_cubemap(face_x_src_gnomonic[available_list].flatten(
            ), face_y_src_gnomonic[available_list].flatten(), weight_type, np.stack((face_flow_x, face_flow_y), axis=1))
            weight_type = "image_warp_error"
            face_weight_mat_2 = projection.get_blend_weight_cubemap(face_erp_x[available_list], face_erp_y[available_list], weight_type,
                                                                    np.stack((face_x_tar_available, face_y_tar_available), axis=1), image_erp_src, image_erp_tar)
            face_weight_mat = np.multiply(face_weight_mat_1, face_weight_mat_2)
        else:
            weight_type = "straightforward"
            face_weight_mat = projection.get_blend_weight_cubemap(face_x_src_gnomonic[available_list].flatten(), face_y_src_gnomonic[available_list].flatten(), weight_type)

        # # for debug weight
        # if not flow_index == 1:
        #     import image_io
        #     temp = np.zeros(face_x_src_gnomonic.shape, np.float)
        #     temp[available_list] = face_weight_mat
        #     image_io.image_show(temp)

        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 0] += face_flow_u * face_weight_mat
        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 1] += face_flow_v * face_weight_mat
        erp_flow_weight_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), np.float) # debug
    non_zero_weight_list = erp_flow_weight_mat != 0
    if not np.all(non_zero_weight_list):
        log.warn("the optical flow weight matrix contain 0.")
    for channel_index in range(0, 2):
        erp_flow_mat[:, :, channel_index][non_zero_weight_list] = erp_flow_mat[:, :, channel_index][non_zero_weight_list] / erp_flow_weight_mat[non_zero_weight_list]

    # TODO poseprocess : bilateral filter
    return erp_flow_mat


def cubemap2erp_depth(cubemap_depth_list, erp_depthmap_height=None, padding_size=0.0):
    """Stitch cubemap's 6 face depth map together.

    :param cubemap_depth_list: the 6 depth map for each face.
    :type cubemap_depth_list: list
    :param erp_depthmap_height: the output ERP image height, defaults to None
    :type erp_depthmap_height: float, optional
    :param padding_size: the face's padding size in gnomonic coordinate, defaults to 0.0
    :type padding_size: float, optional
    :return: ERP image's depth map
    :rtype: numpy
    """
    # check the face images number
    if not 6 == len(cubemap_depth_list):
        raise RuntimeError("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_depth_list[0])[0]
    erp_image_width = None
    if erp_depthmap_height is None:
        erp_image_width = int(cubemap_image_size * 4.0)

    erp_image_height = int(erp_image_width * 0.5)
    if len(cubemap_depth_list[0].shape) > 2:
        log.warn("There are more than 1 channel in the Cubemap depth map, just use the first channel.")

    erp_depthmap = np.zeros((erp_image_height, erp_image_width), dtype=np.float64)
    # Default set the first face's depth image as the reference depth map, the available pixels 2nd channel is 1.
    erp_depthmap_reference = np.zeros((erp_image_height, erp_image_width, 2),  dtype=np.float64)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]
    # gnomonic2image_ratio = (cubemap_image_size - 1) / (2.0 + padding_size * 2.0)

    # convert the perspective depth map to radian distance.
    log.info("Convert the perspective depth map to radina depth map.")
    pbc = 1.0 + padding_size  # projection_boundary_coefficient
    x_grid = np.linspace(-pbc, pbc, cubemap_image_size)
    y_grid = np.linspace(pbc, -pbc, cubemap_image_size)
    gnomonic_coord_x, gnomonic_coord_y = np.meshgrid(x_grid, y_grid)
    cubemap_erp_depth_list = []
    for subemap_depth_persp in cubemap_depth_list:
        center2pixel_length = np.sqrt(np.square(gnomonic_coord_x) + np.square(gnomonic_coord_y) + np.ones_like(gnomonic_coord_y))
        subimage_depthmap_erp = subemap_depth_persp * center2pixel_length
        cubemap_erp_depth_list.append(subimage_depthmap_erp)

    # stitch cubemap depth
    for face_index in range(0, 6):
        # get the tangent ERP image pixel's spherical coordinate location
        # the spherical coordinate range for each face
        face_theta_min = np.amin(face_erp_range_sphere_list[face_index][:, 0], axis=0)
        face_theta_max = np.amax(face_erp_range_sphere_list[face_index][:, 0], axis=0)
        face_phi_min = np.amin(face_erp_range_sphere_list[face_index][:, 1], axis=0)
        face_phi_max = np.amax(face_erp_range_sphere_list[face_index][:, 1], axis=0)

        face_erp_x_min, face_erp_y_max = spherical_coordinates.sph2erp(face_theta_min, face_phi_min, erp_image_height, False)
        face_erp_x_max, face_erp_y_min = spherical_coordinates.sph2erp(face_theta_max, face_phi_max, erp_image_height, False)

        # process the image boundary
        face_erp_x_min = int(face_erp_x_min) if int(face_erp_x_min) > 0 else int(face_erp_x_min - 0.5)
        face_erp_x_max = int(face_erp_x_max + 0.5) if int(face_erp_x_max) > 0 else int(face_erp_x_max)
        face_erp_y_min = int(face_erp_y_min) if int(face_erp_y_min) > 0 else int(face_erp_y_min - 0.5)
        face_erp_y_max = int(face_erp_y_max + 0.5) if int(face_erp_y_max) > 0 else int(face_erp_y_max)

        # 1) get ERP image's pixels corresponding tangent image location
        face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
        face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
        face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
        face_erp_x = np.remainder(face_erp_x, erp_image_width)
        face_erp_y = np.remainder(face_erp_y, erp_image_height)
        face_theta_, face_phi_ = spherical_coordinates.erp2sph((face_erp_x, face_erp_y), erp_image_height, False)

        # 2) get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        center_point = tangent_points_list[face_index]
        theta_0 = center_point[0]
        phi_0 = center_point[1]
        face_x, face_y = gnomonic_projection.gnomonic_projection(face_theta_, face_phi_, theta_0, phi_0)
        inside_list = gnomonic_projection.inside_polygon_2d(np.stack((face_x.flatten(), face_y.flatten()), axis=1),
                                                            np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]]), True).reshape(np.shape(face_x))

        # remove the pixels outside the tangent image & translate to tangent image pixel coordinate,
        # map the gnomonic coordinate to tangent image's pixel coordinate.
        tangent_gnomonic_range = [-pbc, +pbc, -pbc, +pbc]
        face_x_available, face_y_available = gnomonic_projection.gnomonic2pixel(face_x[inside_list], face_y[inside_list],
                                                       0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # 3) offset current face depth map to fit the references depth distribution.
        # copy the reference ERP depth map to reference
        face_erp_x = face_erp_x[inside_list].astype(np.int64)
        face_erp_y = face_erp_y[inside_list].astype(np.int64)

        erp_depthmap_face = np.full((erp_image_height, erp_image_width), 0,  dtype=np.float64)
        erp_depthmap_face[face_erp_y, face_erp_x] = ndimage.map_coordinates(cubemap_erp_depth_list[face_index][:, :], [face_y_available, face_x_available], order=1, mode='constant', cval=0.0)

        if padding_size != 0 and face_index == 0:
            erp_depthmap_reference[face_erp_y, face_erp_x, 1] = 1
            erp_depthmap_reference[face_erp_y, face_erp_x, 0] = ndimage.map_coordinates(cubemap_erp_depth_list[face_index][:, :], [face_y_available, face_x_available], order=1, mode='constant', cval=0.0)

        # compute the scale and offset, scaled them before stitch them.
        face2reference_function = None
        if padding_size != 0 and face_index != 0:
            # 1) extract the overlap are from the reference depth map
            overlap_reference_index = erp_depthmap_reference[:, :, 1] == 1
            overlap_face_index = np.full(overlap_reference_index.shape, False, dtype=bool)
            overlap_face_index[face_erp_y, face_erp_x] = True
            overlap_index = np.logical_and(overlap_reference_index, overlap_face_index)

            if not overlap_index.any():
                # doesn't have any overlap area
                continue

            # 2) fitting the parameters
            overlap_reference_data = erp_depthmap_reference[:, :, 0][overlap_index]
            overlap_face_data = erp_depthmap_face[overlap_index]
            from scipy.interpolate import interp1d
            face2reference_function = interp1d(overlap_face_data, overlap_reference_data, bounds_error=False, fill_value="extrapolate")

        # TODO more padding and overlap
        # 4) get the value of interpolations.
        # erp_depthmap_weight = projection.get_blend_weight() # exclude the padding area
        if face2reference_function is None:
            # TODO padding size 0, have overleap area
            erp_depthmap[face_erp_y, face_erp_x] = erp_depthmap_face[face_erp_y, face_erp_x]
        else:
            erp_depthmap[face_erp_y, face_erp_x] += face2reference_function(erp_depthmap_face[face_erp_y, face_erp_x])

    return erp_depthmap
