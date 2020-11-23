
import numpy as np
from numpy.lib.financial import ipmt

import image_io

import nfov

"""
Implement the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""


def inside_polygon_single(x, y, poly):
    """
    Check if a point is inside a given polygon.
    Reference: http://www.ariel.com.au/a/python-point-int-poly.html

    :param x:
    :param y:
    :param poly:  a list of (x,y) pairs.
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def inside_polygon_2d(points_list, polygon_points, on_segment=True):
    """
    Find out whether the points inside the polygon. 
    The point storage in the numpy array [[x_1, y_1], [x_2, y_2],...[x_n, y_n]].

    :param points: A numpy array including the points locations.
    :param polygon_points: The clock-wise points sequence.
    :param include_boundary: The inside point including the boundary, if True.
    :return: A numpy array fill with Boolean, True mean inside the polygon.
    """
    point_inside = np.full(np.shape(points_list)[0], False, dtype=bool) # the point in the polygon
    point_onsegment = np.full(np.shape(points_list)[0], False, dtype=bool) # the points on the segment

    points_x = points_list[:, 0]
    points_y = points_list[:, 1]

    # try each line segment
    for index in range(np.shape(polygon_points)[0]):
        polygon_1_x = polygon_points[index][0]
        polygon_1_y = polygon_points[index][1]

        polygon_2_x = polygon_points[(index + 1) % len(polygon_points)][0]
        polygon_2_y = polygon_points[(index + 1) % len(polygon_points)][1]

        # on the available Y range
        test_result = np.logical_and(points_y >= min(polygon_1_y, polygon_2_y),
                                     points_y <= max(polygon_1_y, polygon_2_y))
        # on the left of the right eige of bbox
        test_result = np.logical_and(test_result, points_x <= max(polygon_1_x, polygon_2_x))

        if not test_result.any():
            continue

        # get the intersection point
        if polygon_1_y != polygon_2_y:
            intersect_points_x = (points_y[test_result] - polygon_1_y) * \
                (polygon_2_x - polygon_1_x) / (polygon_2_y - polygon_1_y) \
                + polygon_1_x
        else:
            intersect_points_x = points_x[test_result]

        # the poit on the left of segment
        if (points_x[test_result] < intersect_points_x).any():
            point_inside[test_result] = np.logical_not(point_inside[test_result])
        # the point on segment
        if (points_x[test_result] == intersect_points_x).any(): # TODO EPS definition
            onsegment_index = np.full(np.shape(test_result), False, dtype=bool) 
            onsegment_index[test_result] = (points_x[test_result] == intersect_points_x)
            point_onsegment[onsegment_index] = True

    # including the point on the segment line
    if on_segment:
        return np.logical_or(point_inside, point_onsegment)
    else:
        return np.logical_and(point_inside, np.logical_not(point_onsegment))


def inside_triangle(vertex_list, point):
    """
    check whether point in the triangle.
    """
    eps = 10e-7
    ab = vertex_list[1] - vertex_list[0]
    ac = vertex_list[2] - vertex_list[0]
    ap = point - vertex_list[0]

    det = ab[0] * ac[1] - ab[1] * ac[0]
    w_b = (ac[1] * ap[0] - ac[0] * ap[1]) / det
    w_c = (ab[0] * ap[1] - ab[1] * ap[0]) / det
    w_a = 1 - w_b - w_c

    w_a_range = w_a >= -eps and w_a <= 1 + eps
    w_b_range = w_b >= -eps and w_b <= 1 + eps
    w_c_range = w_c >= -eps and w_c <= 1 + eps

    if (w_a_range and w_b_range and w_c_range):
        return True
    else:
        return False


def gnomonic_projection(lambda_, phi, lambda_0, phi_1):
    """
    :param lambda_: longitude, ERP phi
    :param phi: latitude, ERP theta
    :param lambda_0: the center of gnomonic projection
    :param ph_1: 
    """
    cos_c = np.sin(phi_1) * np.sin(phi) + np.cos(phi_1) * np.cos(phi) * np.cos(lambda_ - lambda_0)
    x = np.cos(phi) * np.sin(lambda_ - lambda_0) / cos_c
    y = (np.cos(phi_1) * np.sin(phi) - np.sin(phi_1) * np.cos(phi) * np.cos(lambda_ - lambda_0)) / cos_c
    return x, y


def reverse_gnomonic_projection(x, y, lambda_0, phi_1):
    """
    Reverse gnomonic projection.
    The index of triangle [0-4] is up, [5-9] middle-up, [10-14] middle-down, [15-19] down
    """
    rho = np.sqrt(x**2 + y**2)
    # if rho == 0:
    #     return 0, 0
    c = np.arctan2(rho, 1)

    phi_ = np.arcsin(np.cos(c) * np.sin(phi_1) + (y * np.sin(c) * np.cos(phi_1)) / rho)
    lambda_ = lambda_0 + np.arctan2(x * np.sin(c), rho * np.cos(phi_1) * np.cos(c) - y * np.sin(phi_1) * np.sin(c))

    return lambda_, phi_


if __name__ == "__main__":
    # test inside of polygon 2d
    point_list = np.zeros((8, 2), dtype=np.float64)
    point_list[0, :] = [0, 0]
    point_list[1, :] = [0, 1.3]
    point_list[2, :] = [0.5, -1]
    point_list[3, :] = [-1, 0]
    point_list[4, :] = [1, 0]
    point_list[5, :] = [-1, -1.1]
    point_list[6, :] = [-1, 0.5]
    point_list[7, :] = [1, 0.5]

    polygon_points_list = np.zeros((4, 2), dtype=np.float64)
    polygon_points_list[0, :] = [-1, 1]
    polygon_points_list[1, :] = [1, 1]
    polygon_points_list[2, :] = [1, -1]
    polygon_points_list[3, :] = [-1, -1]

    from functools import reduce
    result = inside_polygon_2d(point_list, polygon_points_list, True)
    print("without on segment", result)
    result_gt = [ True, False,  True,  True,  True, False,  True,  True]
    assert reduce((lambda x, y: x and y), result == result_gt)

    result = inside_polygon_2d(point_list, polygon_points_list, False)
    print("with on segment", result)
    result_gt = [ True, False,  False,  False,  False, False,  False,  False]
    assert reduce((lambda x, y: x and y), result == result_gt)
