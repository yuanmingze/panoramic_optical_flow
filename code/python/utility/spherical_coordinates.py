import numpy as np

import math

from . import image_io


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


def erp2spherical(erp_points):
    """
    convert the point from erp image pixel location to spherical coordinate.
    
    :param erp_points: the point location in ERP image, the x coordinate is in range [0, width), y is in the ranage [0, hight)
    :return: the spherical coordinate points, theta is in the range [-pi, +pi), and phi is in the range [-pi/2, pi/2)
    """
    height = np.shape(erp_points)[1]
    width = np.shape(erp_points)[2]

    if (height * 2) != width:
        raise Exception("the ERP image width {} is not two time of height {}".format(width, height))

    # point location to theta and phi
    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]
    end_points_u = (erp_points_x - width / 2.0) / (width / 2.0) * np.pi
    end_points_v = -(erp_points_y - height / 2.0) / (height / 2.0) * (np.pi / 2.0)

    return np.stack((end_points_u, end_points_v))


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


def car2sph(points_car, points_sph):
    """
    Transform the optical flow from cartesian to spherical coordinate.
    The coordinate system:

    :param points_car: the points array, first column is x, second is y, third is z
    :param points_sph: the points array, theta, phi
    """
    pass


if __name__ == "__main__":
    # test great_circle_distance
    point_pair = []
    point_pair.append([[-np.pi / 2.0, np.pi / 2.0], [-np.pi / 2.0, 0.0]])
    point_pair.append([[-np.pi / 4.0, np.pi / 4.0], [-np.pi / 4.0, -np.pi / 4.0]])
    point_pair.append([[0.0, 0.0], [np.pi / 2.0, 0]])
    point_pair.append([[np.pi / 2.0, 0], [np.pi / 2.0, -np.pi / 4.0]])
    point_pair.append([[0.0, -np.pi / 4.0], [np.pi, -np.pi / 4.0]])
    point_pair.append([[0.0, -np.pi / 2.0], [np.pi / 2.0, -np.pi / 2.0]])
    point_pair.append([[-np.pi * 3.0 / 4.0, 0.0], [np.pi * 3.0 / 4.0, 0.0]])
    point_pair.append([[0.0, 0.0], [0.0, 0.0]])
    point_pair.append([[np.pi / 4.0, np.pi / 4.0], [np.pi / 4.0, np.pi / 4.0]])

    result = [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0, 0.0]

    points_1_u = np.zeros(len(point_pair))
    points_1_v = np.zeros(len(point_pair))
    points_2_u = np.zeros(len(point_pair))
    points_2_v = np.zeros(len(point_pair))
    for index in range(len(point_pair)):
        term = point_pair[index]
        points_1_u[index] = point_pair[index][0][0]
        points_1_v[index] = point_pair[index][0][1]

        points_2_u[index] = point_pair[index][1][0]
        points_2_v[index] = point_pair[index][1][1]

    points_1_u = points_1_u[np.newaxis, ...]
    points_1_v = points_1_v[np.newaxis, ...]
    points_2_u = points_2_u[np.newaxis, ...]
    points_2_v = points_2_v[np.newaxis, ...]

    result_comput = great_circle_distance_uv(points_1_u, points_1_v, points_2_u, points_2_v, radius=1)
    result_comput = result_comput[0]

    for index in range(len(point_pair)):
        print("----{}-----".format(index))
        print("error:    {}".format(np.sqrt(np.abs(result_comput[index] - result[index]))))
        print("GT:       {}".format(result[index]))
        print("Computed: {}".format(result_comput[index]))
