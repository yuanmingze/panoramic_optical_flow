import math
import copy
import numpy as np
from scipy import ndimage
from skimage.transform import resize

import gnomonic_projection as gp
import spherical_coordinates as sc
import polygon
import projection

from logger import Logger

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
    Get the tangent point theta and phi. Known as the theta_0 and phi_0.
    The erp image origin as top-left corner

    :return the tangent face's tangent point and 3 vertices's location.
    """
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # the tangent point
    theta_0 = None
    phi_0 = None

    # the 3 points of tangent triangle in spherical coordinate
    triangle_point_00_theta = None
    triangle_point_00_phi = None
    triangle_point_01_theta = None
    triangle_point_01_phi = None
    triangle_point_02_theta = None
    triangle_point_02_phi = None

    # triangles' row/col range in the erp image
    # erp_image_row_start = None
    # erp_image_row_stop = None
    # erp_image_col_start = None
    # erp_image_col_stop = None

    theta_step = 2.0 * np.pi / 5.0
    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        # tangent point of inscribed spheric
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index * theta_step
        phi_0 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * theta_step
        triangle_point_01_phi = np.pi / 2.0
        triangle_point_02_theta = -np.pi + (triangle_index + 1) * theta_step
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
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        phi_0 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + (triangle_index_temp + 1) * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = -np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
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
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        if triangle_index_temp == 10:
            # cross the ERP image boundary
            triangle_point_00_theta = triangle_point_00_theta + 2 * np.pi
        triangle_point_01_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
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
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_01_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        # cross the ERP image boundary
        if triangle_index_temp == 15:
            triangle_point_01_theta = triangle_point_01_theta + 2 * np.pi
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_02_theta = - np.pi + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.pi / 2.0

        # # spherical coordinate (0,0) is in the center of ERP image
        # erp_image_row_start = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        # erp_image_row_stop = 1.0
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
        # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

    tangent_point = [theta_0, phi_0]

    # the 3 points gnomonic coordinate in tangent image's gnomonic space
    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_theta, triangle_point_00_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_theta, triangle_point_01_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_theta, triangle_point_02_phi, theta_0, phi_0))

    # pading the tangent image
    triangle_points_tangent_pading = polygon.enlarge_polygon(triangle_points_tangent, padding_size)

    # if padding_size != 0.0:
    triangle_points_tangent = copy.deepcopy(triangle_points_tangent_pading)

    # the points in spherical location
    triangle_points_sph = []
    for index in range(3):
        tri_pading_x, tri_pading_y = triangle_points_tangent_pading[index]
        triangle_point_theta, triangle_point_phi = gp.reverse_gnomonic_projection(tri_pading_x, tri_pading_y, theta_0, phi_0)
        triangle_points_sph.append([triangle_point_theta, triangle_point_phi])

    # compute bounding box of the face in spherical coordinate
    availied_sph_area = []
    availied_sph_area = np.array(copy.deepcopy(triangle_points_sph))
    triangle_points_tangent_pading = np.array(triangle_points_tangent_pading)
    point_insert_x = np.sort(triangle_points_tangent_pading[:, 0])[1]
    point_insert_y = np.sort(triangle_points_tangent_pading[:, 1])[1]
    availied_sph_area = np.append(availied_sph_area, [gp.reverse_gnomonic_projection(point_insert_x, point_insert_y, theta_0, phi_0)], axis=0)
    # the bounding box of the face with spherical coordinate
    availied_ERP_area_sph = []  # [min_theta, max_theta, min_phi, max_phi]
    if 0 <= triangle_index <= 4:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.pi / 2.0)
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down
    elif 15 <= triangle_index <= 19:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(-np.pi / 2.0)
    else:
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))

    # else:
    #     triangle_points_sph.append([triangle_point_00_theta, triangle_point_00_theta])
    #     triangle_points_sph.append([triangle_point_01_theta, triangle_point_01_theta])
    #     triangle_points_sph.append([triangle_point_02_theta, triangle_point_02_theta])

    #     availied_ERP_area.append(erp_image_row_start)
    #     availied_ERP_area.append(erp_image_row_stop)
    #     availied_ERP_area.append(erp_image_col_start)
    #     availied_ERP_area.append(erp_image_col_stop)

    return {"tangent_point": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph, "availied_ERP_area": availied_ERP_area_sph}


def erp2ico_image(erp_image, tangent_image_width, padding_size=0.0, full_face_image=False):
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
    if len(erp_image.shape) == 3:
        if np.shape(erp_image)[2] == 4:
            erp_image = erp_image[:, :, 0:3]
    elif len(erp_image.shape) == 2:
        log.info("project single channel disp or depth map")
        erp_image = np.expand_dims(erp_image, axis=2)

    # ERP image size
    erp_image_height = np.shape(erp_image)[0]
    erp_image_width = np.shape(erp_image)[1]
    channel_number = np.shape(erp_image)[2]

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
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
        if not full_face_image:
            gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
            pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
            inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(gnom_range_xv.shape)

        # project to tangent image
        tangent_point = triangle_param["tangent_point"]
        tangent_triangle_theta_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_point[0], tangent_point[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2erp(tangent_triangle_theta_, tangent_triangle_phi_, erp_image_height, wrap_around=True)

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                             0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        if channel_number == 1:
            tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], 255)
        elif channel_number == 3:
            tangent_image = np.full([tangent_image_height, tangent_image_width, 4], 255)
        else:
            log.error("The channel number is {}".format(channel_number))

        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y, tangent_image_x, channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255)

        # set the pixels outside the boundary to transparent
        tangent_image[:, :, 3] = 0
        tangent_image[tangent_image_y, tangent_image_x, 3] = 255
        tangent_image_list.append(tangent_image)

    return tangent_image_list


def ico2erp_image(tangent_images, erp_image_height, padding_size=0.0, blender_method=None):
    """Stitch the level-0 icosahedron's tangent image to ERP image.

    blender_method:
        - None: just sample the triangle area;
        - Mean: the mean value on the overlap area.

    TODO there are seam on the stitched erp image.

    :param tangent_images: 20 tangent images in order.
    :type tangent_images: a list of numpy
    :param erp_image_height: the output erp image's height.
    :type erp_image_height: int
    :param padding_size: the face image's padding size.
    :type padding_size: float
    :param blender_method: the method used to blend sub-images. 
    :type blender_method: str
    :return: the stitched ERP image
    :type numpy
    """
    if len(tangent_images) != 20:
        log.error("The tangent's images triangle number is {}.".format(len(tangent_images)))

    images_channels_number = tangent_images[0].shape[2]
    if images_channels_number == 4:
        log.warn("the face image is RGBA image, convert the output to RGB image.")
        images_channels_number = 3
    erp_image_width = erp_image_height * 2
    erp_image = np.full([erp_image_height, erp_image_width, images_channels_number], 0)

    tangent_image_height = tangent_images[0].shape[0]
    tangent_image_width = tangent_images[0].shape[1]

    erp_weight_mat = np.zeros((erp_image_height, erp_image_width), dtype=np.float64)

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
        theta_0 = triangle_param["tangent_point"][0]
        phi_0 = triangle_param["tangent_point"][1]
        tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], theta_0, phi_0)

        # the pixels in the tangent triangle
        triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])
        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)

        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        pixel_eps = abs(tangent_xv[0, 0] - tangent_xv[0, 1]) / (2 * tangent_image_width)

        if blender_method is None:
            available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                         triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

            # the tangent available gnomonic coordinate sample the pixel from the tangent image
            tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                       0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

            for channel in range(0, images_channels_number):
                erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] = \
                    ndimage.map_coordinates(tangent_images[triangle_index][:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
        elif blender_method == "mean":
            triangle_points_tangent = [[gnomonic_x_min, gnomonic_y_max],
                                       [gnomonic_x_max, gnomonic_y_max],
                                       [gnomonic_x_max, gnomonic_y_min],
                                       [gnomonic_x_min, gnomonic_y_min]]
            available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                         triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

            tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                       0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)
            for channel in range(0, images_channels_number):
                erp_face_image = ndimage.map_coordinates(tangent_images[triangle_index][:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
                erp_image[triangle_yv[available_pixels_list].astype(np.int), triangle_xv[available_pixels_list].astype(np.int), channel] += erp_face_image

            face_weight_mat = np.ones(erp_face_image.shape, np.float64)
            erp_weight_mat[triangle_yv[available_pixels_list].astype(np.int64), triangle_xv[available_pixels_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    if blender_method == "mean":
        # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), np.float) # debug
        non_zero_weight_list = erp_weight_mat != 0
        if not np.all(non_zero_weight_list):
            log.warn("the optical flow weight matrix contain 0.")
        for channel_index in range(0, images_channels_number):
            erp_image[:, :, channel_index][non_zero_weight_list] = erp_image[:, :, channel_index][non_zero_weight_list] / erp_weight_mat[non_zero_weight_list]

    return erp_image


def erp2ico_flow(erp_flow_mat, tangent_image_width, padding_size=0.0, full_face_image=False):
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
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
        if not full_face_image:
            gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
            pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
            inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(np.shape(gnom_range_xv))

        # 0) Get the tangent image pixels' ERP location, convert the ERP optical flow's UV to tangent image's UV
        # flow start point from gnomonic --> spherical coordinate --> pixel location
        tangent_point = triangle_param["tangent_point"]  # tangent center project point
        tangent_triangle_theta_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_point[0], tangent_point[1])
        # TODO check the wrap around implement
        face_erp_pixel_x, face_erp_pixel_y = sc.sph2erp(tangent_triangle_theta_, tangent_triangle_phi_, erp_image_height, wrap_around=False)

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
        face_image_x_target, face_image_y_target = gp.gnomonic_projection(face_pixel_sph[0, :], face_pixel_sph[1, :], tangent_point[0], tangent_point[1])

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


def ico2erp_flow(tangent_flows_list, erp_flow_height=None, padding_size=0.0, image_erp_src=None, image_erp_tar=None, of_wrap_around=False):
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

    # erp flow and blending weight
    erp_flow_mat = np.zeros((erp_flow_height, erp_flow_width, 2), dtype=np.float64)
    erp_flow_weight_mat = np.zeros((erp_flow_height, erp_flow_width), dtype=np.float64)

    for face_index in range(0, len(tangent_flows_list)):
        # for triangle_index in range(0, 2):
        log.debug("stitch the tangent image {}".format(face_index))
        face_param = get_icosahedron_parameters(face_index, padding_size)
        theta_0 = face_param["tangent_point"][0]
        phi_0 = face_param["tangent_point"][1]
        triangle_points_tangent = np.array(face_param["triangle_points_tangent"])
        availied_ERP_area = face_param["availied_ERP_area"]

        # 1) get tangent face available pixles range in ERP spherical coordinate
        erp_flow_col_start, erp_flow_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_flow_height, wrap_around=False)
        erp_flow_col_stop, erp_flow_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_flow_height, wrap_around=False)
        # process the tangent flow boundary
        erp_flow_col_start = int(erp_flow_col_start) if int(erp_flow_col_start) > 0 else int(erp_flow_col_start - 0.5)
        erp_flow_col_stop = int(erp_flow_col_stop + 0.5) if int(erp_flow_col_stop) > 0 else int(erp_flow_col_stop)
        erp_flow_row_start = int(erp_flow_row_start) if int(erp_flow_row_start) > 0 else int(erp_flow_row_start - 0.5)
        erp_flow_row_stop = int(erp_flow_row_stop + 0.5) if int(erp_flow_row_stop) > 0 else int(erp_flow_row_stop)
        triangle_x_range = np.linspace(erp_flow_col_start, erp_flow_col_stop, erp_flow_col_stop - erp_flow_col_start + 1)
        triangle_y_range = np.linspace(erp_flow_row_start, erp_flow_row_stop, erp_flow_row_stop - erp_flow_row_start + 1)

        face_src_x_erp, face_src_y_erp = np.meshgrid(triangle_x_range, triangle_y_range)
        face_src_x_erp = np.remainder(face_src_x_erp, erp_flow_width)  # process the wrap around
        face_src_y_erp = np.remainder(face_src_y_erp, erp_flow_height)

        # 2) get the pixels location in tangent image location
        # ERP image space --> spherical space
        face_src_xy_sph = sc.erp2sph((face_src_x_erp, face_src_y_erp), erp_flow_height, False)

        # spherical space --> normailzed tangent image space
        face_src_x_gnom, face_src_y_gnom = gp.gnomonic_projection(face_src_xy_sph[0, :, :], face_src_xy_sph[1, :, :], theta_0, phi_0)

        # the available (in the triangle) pixels list
        pixel_eps = abs(face_src_x_gnom[0, 0] - face_src_x_gnom[0, 1]) / (2 * tangent_flow_width)
        available_list = gp.inside_polygon_2d(np.stack((face_src_x_gnom.flatten(), face_src_y_gnom.flatten()), axis=1), triangle_points_tangent, on_line=True, eps=pixel_eps)
        available_list = available_list.reshape(face_src_x_gnom.shape)

        # normailzed tangent image space --> tangent image space
        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)
        face_src_range_gnom = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        face_src_x_gnom_pixel, face_src_y_gnom_pixel = gp.gnomonic2pixel(face_src_x_gnom[available_list], face_src_y_gnom[available_list], 0.0, tangent_flow_width, tangent_flow_height, face_src_range_gnom)

        # 3) get the value of optical flow end point location
        # 3-0) get the tangent images flow in the tangent image space, ignore the pixels outside the tangent image
        face_flow_u_gnom = ndimage.map_coordinates(tangent_flows_list[face_index][:, :, 0], [face_src_y_gnom_pixel, face_src_x_gnom_pixel], order=1, mode='constant', cval=255)
        face_tar_x_gnom_pixel_avail = face_src_x_gnom_pixel + face_flow_u_gnom

        face_flow_v_gnom = ndimage.map_coordinates(tangent_flows_list[face_index][:, :, 1], [face_src_y_gnom_pixel, face_src_x_gnom_pixel], order=1, mode='constant', cval=255)
        face_tar_y_gnom_pixel_avail = face_src_y_gnom_pixel + face_flow_v_gnom

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        face_tar_x_gnom_avail, face_tar_y_gnom_avail = gp.pixel2gnomonic(face_tar_x_gnom_pixel_avail, face_tar_y_gnom_pixel_avail, 0.0,
                                                                     tangent_flow_width, tangent_flow_height, face_src_range_gnom)
        # tangent normailzed space --> spherical space
        face_tar_x_sph_avail, face_tar_y_sph_avail = gp.reverse_gnomonic_projection(face_tar_x_gnom_avail, face_tar_y_gnom_avail, theta_0, phi_0)

        # 3-2) process the optical flow wrap-around, including face, use the shorted path as real path.
        if of_wrap_around:
            log.info("ERP optical flow with wrap around. Face index {}".format(face_index))
            face_src_x_sph_avail = face_src_xy_sph[0, :, :][available_list]
            cross_boundary = np.abs(face_src_x_sph_avail - face_tar_x_sph_avail) > np.pi
            cross_x_axis_minus2pi = np.logical_and(cross_boundary, face_src_x_sph_avail < 0)
            cross_x_axis_plus2pi = np.logical_and(cross_boundary, face_src_x_sph_avail >= 0)
            face_tar_x_sph_avail[cross_x_axis_minus2pi] = face_tar_x_sph_avail[cross_x_axis_minus2pi] - np.pi * 2
            face_tar_x_sph_avail[cross_x_axis_plus2pi] = face_tar_x_sph_avail[cross_x_axis_plus2pi] + np.pi * 2
            # spherical space --> ERP image space
            face_tar_x_erp, face_tar_y_erp = sc.sph2erp(face_tar_x_sph_avail, face_tar_y_sph_avail, erp_flow_height, False)
        else:
            face_tar_x_erp, face_tar_y_erp = sc.sph2erp(face_tar_x_sph_avail, face_tar_y_sph_avail, erp_flow_height, True)

        # 4) get ERP flow with source and target pixels location
        # 4-0) the ERP flow
        face_flow_u_erp = face_tar_x_erp - face_src_x_erp[available_list]
        face_flow_v_erp = face_tar_y_erp - face_src_y_erp[available_list]

        # 4-1) compute the all available pixels' weight to blend the optical flow
        face_weight_mat = np.ones(face_src_y_gnom_pixel.shape, dtype= np.float64)

        # weight_type = "straightforward"
        # triangle_points_tangent_weight = get_icosahedron_parameters(triangle_index, 0.0)["triangle_points_tangent"]
        # face_weight_mat = projection.get_blend_weight(tangent_xv_gnom[available_list].flatten(), tangent_yv_gnom[available_list].flatten(), weight_type, np.stack((face_flow_x, face_flow_y), axis=1), gnomonic_bounding_box = triangle_points_tangent_weight)

        # # resize the erp image
        # if image_erp_src.shape[:2] != [erp_flow_height, erp_flow_width]:
        #     image_erp_src = resize(image_erp_src, (erp_flow_height, erp_flow_width)) * 255.0
        # if image_erp_tar.shape[:2] != [erp_flow_height, erp_flow_width]:
        #     image_erp_tar = resize(image_erp_tar, (erp_flow_height, erp_flow_width)) * 255.0
        # face_weight_mat_1 = projection.get_blend_weight_ico(tangent_xv_gnom[available_list].flatten(), tangent_yv_gnom[available_list].flatten(),
        #                                                     "normal_distribution_flowcenter", np.stack((face_flow_x, face_flow_y), axis=1),
        #                                                     gnomonic_bounding_box=triangle_points_tangent)
        # face_weight_mat_2 = projection.get_blend_weight_ico(triangle_xv[available_list], triangle_yv[available_list],
        #                                                     "image_warp_error", np.stack((tangent_xv_tar_pixel, tangent_yv_tar_pixel), axis=1),
        #                                                     image_erp_src, image_erp_tar)
        # face_weight_mat = face_weight_mat_1 * face_weight_mat_2

        # # for debug weight
        # if triangle_index == -1:
        #     from . import image_io
        #     temp = np.zeros(tangent_xv_gnom.shape, np.float)
        #     temp[available_list] = face_weight_mat
        #     image_io.image_show(temp)

        # blender ERP flow and weight
        erp_flow_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64), 0] += face_flow_u_erp * face_weight_mat
        erp_flow_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64), 1] += face_flow_v_erp * face_weight_mat

        erp_flow_weight_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), np.float)
    # erp_flow_weight_mat = np.where(erp_flow_weight_mat < 1, erp_flow_weight_mat, 0)
    # import image_io
    # image_io.image_show(erp_flow_weight_mat)

    non_zero_weight_list = erp_flow_weight_mat != 0.0
    if not np.all(non_zero_weight_list):
        log.warn("the optical flow weight matrix contain 0.")
    for channel_index in range(0, 2):
        erp_flow_mat[:, :, channel_index][non_zero_weight_list] = erp_flow_mat[:, :, channel_index][non_zero_weight_list] / erp_flow_weight_mat[non_zero_weight_list]

    return erp_flow_mat
