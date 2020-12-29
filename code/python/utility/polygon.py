import numpy as np
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def find_intersection(p1,  p2,  p3,  p4):
    """Find the point of intersection between two line.
    
    The two lines are p1 --> p2 and p3 --> p4.
    Reference:http://csharphelper.com/blog/2020/12/enlarge-a-polygon-that-has-colinear-vertices-in-c/

    :param p1: line 1's start point
    :type p1: list
    :param p2: line 1's end point
    :type p2: list
    :param p3: line 2's start point
    :type p3: list
    :param p4: line 2's end point
    :type p4: list
    :return: The intersection point of two line
    :rtype: list
    """
    # the segments
    dx12 = p2[0] - p1[0]
    dy12 = p2[1] - p1[1]
    dx34 = p4[0] - p3[0]
    dy34 = p4[1] - p3[1]

    denominator = (dy12 * dx34 - dx12 * dy34)
    if denominator == 0:
        # The two lines are parallel
        return None

    t1 = ((p1[0] - p3[0]) * dy34 + (p3[1] - p1[1]) * dx34) / denominator
    t2 = ((p3[0] - p1[0]) * dy12 + (p1[1] - p3[1]) * dx12) / -denominator

    # point of intersection.
    intersection = [p1[0] + dx12 * t1, p1[1] + dy12 * t1]
    return intersection


def enlarge_polygon(old_points, offset):
    """Return points representing an enlarged polygon.

    Reference: http://csharphelper.com/blog/2016/01/enlarge-a-polygon-in-c/
    
    :param old_points: the points should be in clock wise
    :type: list [[x_1,y_1], [x_2, y_2]......]
    :param offset: the ratio of the polygon enlarged
    :type: float
    :return: the offset points
    :rtype: list
    """
    enlarged_points = []
    num_points = len(old_points)
    for j in range(num_points):
        # 0) find "out" side
        # // TODO check whether is clockwise?
        # https://stackoverflow.com/questions/3749678/expand-fill-of-convex-polygon

        # the points before and after j.
        i = (j - 1)
        if i < 0:
            i += num_points
        k = (j + 1) % num_points

        # 1) Move the points by the offset.
        # the points of line parallel to ij
        v1 = np.array([old_points[j][0] - old_points[i][0], old_points[j][1] - old_points[i][1]], np.float)
        norm = np.linalg.norm(v1)
        v1 = v1 / norm * offset
        n1 = [-v1[1], v1[0]]
        pij1 = [old_points[i][0] + n1[0], old_points[i][1] + n1[1]]
        pij2 = [old_points[j][0] + n1[0], old_points[j][1] + n1[1]]

        # the points of line parallel to jk
        v2 = np.array([old_points[k][0] - old_points[j][0], old_points[k][1] - old_points[j][1]], np.float)
        norm = np.linalg.norm(v2)
        v2 = v2 / norm * offset
        n2 = [-v2[1], v2[0]]
        pjk1 = [old_points[j][0] + n2[0], old_points[j][1] + n2[1]]
        pjk2 = [old_points[k][0] + n2[0], old_points[k][1] + n2[1]]

        # 2) get the shifted lines ij and jk intersect
        lines_intersect = find_intersection(pij1, pij2, pjk1, pjk2)
        enlarged_points.append(lines_intersect)

    return enlarged_points
