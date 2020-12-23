import numpy as np
import math


"""
The optical flow U is corresponding phi, and V is corresponding theta.
Conversion: 1) the arguments order is phi(U or X) and theta(V or Y)
            2) the optical flow layers order are U and V
"""


def great_circle_distance(points_1, points_2, radius=1):
    """
    Get the distance between two points in the sphere, the grate-circle distance.
    Reference: https://en.wikipedia.org/wiki/Great-circle_distance.

    :param first: the numpy array [lambda_1, phi_1]
    :param second: the numpy array [lambda_2, phi_2]
    :param radius: the radius, the default is 1
    :return distance: the great-circle distance of two point.
    """
    return great_circle_distance_uv(points_1[0], points_1[1], points_2[0], points_2[1], radius)


def great_circle_distance_uv(points_1_phi, points_1_theta, points_2_phi, points_2_theta, radius=1):
    """
    Reference: 

    :param points_1_theta:
    :param points_1_phi:
    :param points_2_theta:
    :param points_2_phi:
    """
    # # compute great circle distance
    # phi_diff = np.abs(points_1_phi - points_2_phi)
    # param_1 = np.sin(points_1_theta) * np.sin(points_2_theta)
    # param_2 = np.cos(points_1_theta) * np.cos(points_2_theta)
    # central_angle_delta = np.arccos(param_1 + param_2 * np.cos(phi_diff))

    # HaversineDistance
    lat1 = points_1_theta
    lon1 = points_1_phi
    lat2 = points_2_theta
    lon2 = points_2_phi
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat1) \
        * np.cos(lat2) * np.sin(dlon/2) * np.sin(dlon/2)
    central_angle_delta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    if np.isnan(central_angle_delta).any():
        raise Exception("the angle contain NAN")

    return radius * central_angle_delta


def get_angle(points_A, points_B, points_C):
    """
    :param points_A:
    :param points_B:
    :param points_C:
    """
    points_A_phi = points_A[0]
    points_A_theta = points_A[1]
    points_B_phi = points_B[0]
    points_B_theta = points_B[1]
    points_C_phi = points_C[0]
    points_C_theta = points_C[1]
    return get_angle_uv(points_A_phi, points_A_theta, points_B_phi, points_B_theta, points_C_phi, points_C_theta)


def get_angle_uv(points_A_phi, points_A_theta,
                 points_B_phi, points_B_theta,
                 points_C_phi, points_C_theta):
    """
    Spherical trigonometry, \cos a= \cos b \cos c + \sin b \sin c \cos A
    Reference: https://en.wikipedia.org/wiki/Spherical_trigonometry
               https://mathworld.wolfram.com/SphericalTrigonometry.html

    :param points_A_theta:
    :param points_A_phi:
    :param points_B_theta:
    :param points_B_phi:
    :param points_C_theta:
    :param points_C_phi:
    :return: the angle between AB and AC
    """
    length_AB = great_circle_distance_uv(points_A_phi, points_A_theta, points_B_phi, points_B_theta, radius=1)
    length_AC = great_circle_distance_uv(points_A_phi, points_A_theta, points_C_phi, points_C_theta, radius=1)
    length_BC = great_circle_distance_uv(points_B_phi, points_B_theta, points_C_phi, points_C_theta, radius=1)

    # # with the arccos
    # data = (np.cos(length_BC) - np.cos(length_AC) * np.cos(length_AB)) / (np.sin(length_AC) * np.sin(length_AB))
    # # remove the float error
    # if (np.abs(data) > (1.0 + np.finfo(float).eps * 100)).any():
    #     raise RuntimeError("the angle_A is not in range [-1,1]")
    # else:
    #     data[data > 1.0] = 1.0
    #     data[data < -1.0] = -1.0
    # angle_A = np.arccos(data) - np.pi

    # with tangent two avoid the error of Float
    s = 0.5 * (length_BC + length_AC + length_AB)
    numerator = np.sin(s-length_AC) * np.sin(s-length_AB)
    denominator = np.sin(s) * np.sin(s-length_BC)
    #denominator[denominator == 0] = pow(0.1, 10)
    temp_data = numerator / denominator
    # angle_sign = np.sign(temp_data)
    angle_A = 2 * np.arctan(np.sqrt(np.abs(temp_data)))

    if np.isnan(angle_A).any():
        raise Exception("the angle_A contain NAN")

    return angle_A


def erp2spherical(erp_points, erp_image_height=None, wrap_around=False):
    """
    ERP image Original is top_left, spherical coordinate origin as center.
    convert the point from erp image pixel location to spherical coordinate.
    the point location in ERP image, the x coordinate is in range [0, width), y is in the ranage [0, hight).
    The first pixel 0 is corresponding azimuth -PI, and the last pixel image_width - 1 is corresponding (2PI) / image_width * (image_width -1 - 0.5* image_width). 

    :param erp_points: the point location in ERP image, the x coordinate is in range [0, width), y is in the ranage [0, hight)
    :type erp_points: numpy
    :param erp_image_height: ERP image's height, defaults to None
    :type erp_image_height: int, optional
    :param wrap_around: if true, process the input points wrap around to make all point's x and y in the range [-pi,+pi], [-pi/2, +pi/2]
    :type wrap_around: bool
    :return: the spherical coordinate points, theta is in the range [-pi, +pi), and phi is in the range [-pi/2, pi/2)
    :rtype: numpy
    """
    # 0) the ERP image size
    if erp_image_height == None:
        height = np.shape(erp_points)[1]
        width = np.shape(erp_points)[2]

        if (height * 2) != width:
            raise Exception("the ERP image width {} is not two time of height {}".format(width, height))
    else:
        height = erp_image_height
        width = height * 2

    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]
    if wrap_around:
        erp_points_x = np.remainder(erp_points_x, width)
        erp_points_y = np.remainder(erp_points_x, height)

    # 1) point location to theta and phi
    end_points_u = (erp_points_x - width / 2.0) / (width / 2.0) * np.pi
    end_points_v = -(erp_points_y - height / 2.0) / (height / 2.0) * (np.pi / 2.0)
    return np.stack((end_points_u, end_points_v))


def spherical2epr(phi, theta, image_height, wrap_around=False):
    """ Transform the spherical coordinate location to ERP image pixel location.
    The range of erp phi is [-pi, +pi), theta is [-0.5*pi, +0.5*pi].
    The origin of the ERP is in the Top-Left, and origin of the spherical at the center of ERP image.

    :param phi: longitude 
    :type phi: numpy
    :param theta: latitude
    :type theta: numpy
    :param image_height: the height of the ERP image. the image width is 2 times of image height
    :type image_height: [type]
    :param wrap_around: if yes process the wrap around case, if no do not.
    :type wrap_around: bool, optional
    :return: the pixel location in the ERP image.
    :rtype: numpy
    """
    x = (phi + np.pi) / (2.0 * np.pi) * (2 * image_height)
    y = -(theta - 0.5 * np.pi) / np.pi * image_height

    # process the wrap around case
    if wrap_around:
        x = np.remainder(x, image_height * 2)
        y = np.remainder(y, image_height)
    return x, y


def flow_warp_meshgrid(motion_flow_u, motion_flow_v):
    """
    warp the the original points with the motion vector, meanwhile process the warp around.

    :return: the target points
    """
    if np.shape(motion_flow_u) != np.shape(motion_flow_v):
        raise Exception("motion flow u shape {} is not equal motion flow v shape {}".format(np.shape(motion_flow_u), np.shape(motion_flow_v)))

    # get the mesh grid
    height = np.shape(motion_flow_u)[0]
    width = np.shape(motion_flow_u)[1]
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    x_array, y_array = np.meshgrid(x_index, y_index)

    # get end point location
    end_points_u = x_array + motion_flow_u
    end_points_v = y_array + motion_flow_v

    # process the warp around
    end_points_u[end_points_u >= width] = end_points_u[end_points_u >= width] - width
    end_points_u[end_points_u < 0] = end_points_u[end_points_u < 0] + width

    end_points_v[end_points_v >= height] = end_points_v[end_points_v >= height] - height
    end_points_v[end_points_v < 0] = end_points_v[end_points_v < 0] + height

    return np.stack((end_points_u, end_points_v))


def car2sph(points_car, points_sph, min_radius=1e-10):
    """
    Transform the 3D point from cartesian to unit spherical coordinate.

    Right hand coordinate system: 
    The 3D coordinate system is right hand, X is back, Y is right, Z is up. 

    :param points_car: The 3D point array, is [point_number, 3], first column is x, second is y, third is z
    :type points_car: numpy
    :return: the points spherical coordinate., [azimuth, polar] (theta, phi)
    :rtype: numpy
    """
    radius = np.linalg.norm(points_car, axis=1)

    valid_list = radius > min_radius  # set the 0 radius to origin.

    azimuth = np.zeros((points_car.shape[0]), np.float)
    azimuth[valid_list] = np.arctan2(points_car[:, 1][valid_list], -points_car[:, 0][valid_list])

    polar = np.zeros((points_car.shape[0]), np.float)
    polar[valid_list] = np.arcsin(np.divide(points_car[:, 2][valid_list], radius[valid_list]))

    return np.stack((azimuth, polar), axis=1)


def sph2car(phi, theta, radius=1.0):
    """Transform the spherical coordinate to cartesian 3D point.

    :param phi: longitude
    :type phi: numpy
    :param theta: latitude
    :type theta: numpy
    :param radius: the radius of projection sphere
    :type radius: float
    :return: +x right, +y down, +z is froward
    :rtype: numpy
    """
    # points_cartesian_3d = np.array.zeros((phi.shape[0],3),np.float)
    x = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.cos(theta) * np.cos(phi)
    y = -radius * np.sin(theta)

    return np.stack((x, y, z), axis=1)
