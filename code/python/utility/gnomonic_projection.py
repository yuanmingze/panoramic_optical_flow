
import ipdb
import numpy as np
from numpy.lib.financial import ipmt

import image_io

import nfov

"""
Implement the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""
# def inside_triangle(vertex_list, point, eps = 10e-7):
#     """
#     check whether point in the triangle.
#     """s
#     ab = vertex_list[1] - vertex_list[0]
#     ac = vertex_list[2] - vertex_list[0]
#     ap = point - vertex_list[0]

#     det = ab[0] * ac[1] - ab[1] * ac[0]
#     w_b = (ac[1] * ap[0] - ac[0] * ap[1]) / det
#     w_c = (ab[0] * ap[1] - ab[1] * ap[0]) / det
#     w_a = 1 - w_b - w_c

#     w_a_range = w_a >= -eps and w_a <= 1 + eps
#     w_b_range = w_b >= -eps and w_b <= 1 + eps
#     w_c_range = w_c >= -eps and w_c <= 1 + eps

#     if (w_a_range and w_b_range and w_c_range):
#         return True
#     else:
#         return False


# def inside_polygon_single(x, y, poly):
#     """
#     Check if a point is inside a given polygon.
#     Reference: http://www.ariel.com.au/a/python-point-int-poly.html

#     :param x:
#     :param y:
#     :param poly:  a list of (x,y) pairs.
#     """
#     n = len(poly)
#     inside = False

#     p1x, p1y = poly[0]
#     for i in range(n+1):
#         p2x, p2y = poly[i % n]
#         if y > min(p1y, p2y):
#             if y <= max(p1y, p2y):
#                 if x <= max(p1x, p2x):
#                     if p1y != p2y:
#                         xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
#                     if p1x == p2x or x <= xinters:
#                         inside = not inside
#         p1x, p1y = p2x, p2y

#     return inside


def inside_polygon_2d(points_list, polygon_points, on_line=False, eps=1e-4):
    """ Find out whether the points inside the polygon. 
    Implement PIP method. If 
    The point storage in the numpy array [[x_1, y_1], [x_2, y_2],...[x_n, y_n]].

    :param points_list: A numpy array including the points locations.
    :type points_list: numpy
    :param polygon_points:  The clock-wise points sequence.
    :type polygon_points: numpy
    :param on_line: The inside point including the boundary, if True. defaults to False
    :type on_line: bool, optional
    :param eps: Use the set the polygon's line width. The distance between two pixel. defaults to 1e-4
    :type eps: float, optional
    :return: A numpy array fill with Boolean, True mean inside the polygon.
    :rtype: numpy
    """
    point_inside = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point in the polygon
    online_index = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point on the polygon lines

    points_x = points_list[:, 0]
    points_y = points_list[:, 1]

    def GREATER(a, b): return a >= b
    def LESS(a, b): return a <= b

    # try each line segment
    for index in range(np.shape(polygon_points)[0]):
        polygon_1_x = polygon_points[index][0]
        polygon_1_y = polygon_points[index][1]

        polygon_2_x = polygon_points[(index + 1) % len(polygon_points)][0]
        polygon_2_y = polygon_points[(index + 1) % len(polygon_points)][1]

        # exist points on the available XY range
        test_result = np.logical_and(GREATER(points_y, min(polygon_1_y, polygon_2_y)), LESS(points_y, max(polygon_1_y, polygon_2_y)))
        test_result = np.logical_and(test_result, LESS(points_x, max(polygon_1_x, polygon_2_x)))
        if not test_result.any():
            continue

        # get the intersection point
        if abs(polygon_1_y - polygon_2_y) < eps:
            test_result = np.logical_and(test_result, GREATER(points_x, min(polygon_1_x, polygon_2_x)))
            intersect_points_x = points_x[test_result]
        else:
            intersect_points_x = (points_y[test_result] - polygon_1_y) * \
                (polygon_2_x - polygon_1_x) / (polygon_2_y - polygon_1_y) + polygon_1_x

        # the point on the line
        on_line_list = LESS(abs(points_x[test_result] - intersect_points_x), eps)
        if on_line_list.any():
            online_index[test_result] = np.logical_or(online_index[test_result], on_line_list)

        # the point on the left of the line
        if LESS(points_x[test_result], intersect_points_x).any():
            test_result[test_result] = np.logical_and(test_result[test_result], LESS(points_x[test_result], intersect_points_x))
            point_inside[test_result] = np.logical_not(point_inside[test_result])

    if on_line:
        return np.logical_or(point_inside, online_index)
    else:
        return np.logical_and(point_inside, np.logical_not(online_index))


def gnomonic_projection(lambda_, phi, lambda_0, phi_1):
    """
    Convet point form the spherical coordinate to tangent image's coordinate.

    :param lambda_: longitude, ERP phi
    :param phi: latitude, ERP theta
    :param lambda_0: the center of gnomonic projection
    :param ph_1: 
    
    :return the point's coordinate in this tangent image
    """
    cos_c = np.sin(phi_1) * np.sin(phi) + np.cos(phi_1) * np.cos(phi) * np.cos(lambda_ - lambda_0)
    x = np.cos(phi) * np.sin(lambda_ - lambda_0) / cos_c
    y = (np.cos(phi_1) * np.sin(phi) - np.sin(phi_1) * np.cos(phi) * np.cos(lambda_ - lambda_0)) / cos_c
    return x, y


def reverse_gnomonic_projection(x, y, lambda_0, phi_1):
    """ 
    Reverse gnomonic projection. Get tangent image's point spherical coordinate.
    The index of triangle [0-4] is up, [5-9] middle-up, [10-14] middle-down, [15-19] down

    :param x: [description]
    :type x: [type]
    :param y: [description]
    :type y: [type]
    :param lambda_0: [description]
    :type lambda_0: [type]
    :param phi_1: [description]
    :type phi_1: [type]
    :return: [description]
    :rtype: [type]
    """
    rho = np.sqrt(x**2 + y**2)
    # if rho == 0:
    #     return 0, 0
    c = np.arctan2(rho, 1)

    phi_ = np.arcsin(np.cos(c) * np.sin(phi_1) + (y * np.sin(c) * np.cos(phi_1)) / rho)
    lambda_ = lambda_0 + np.arctan2(x * np.sin(c), rho * np.cos(phi_1) * np.cos(c) - y * np.sin(phi_1) * np.sin(c))

    return lambda_, phi_
