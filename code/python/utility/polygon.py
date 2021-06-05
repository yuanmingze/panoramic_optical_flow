import numpy as np
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def detect_intersection_segments_array(p1_, p2_, p3, p4, including_online=False):
    """
    1st segment p1->p2, 2nd segment p3->p4.

    :param p1: 1st segment 1st points (x,y).
    :type p1: lists
    :param p2: 1st segment 2nd points (x,y).
    :type p2: lists
    :param p3: 2nd segment 1st point. [2, points_number]
    :type p3: numpy
    :param p4: 2nd segment 2nd point. [2, points_number]
    :type p4: numpy
    :return: 2 given segments intersect or not, returns true if the segment 'p1p2' and 'p3p4' intersect.
    :rtype: numpy bool
    """
    def orientation(p, q, r):
        """ to find the orientation of an ordered triplet (p,q,r), 3 points (x,y)
        function returns the following values:
            0 : Colinear points
            1 : Clockwise points
            2 : Counterclockwise
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        """
        val = ((q[1, :] - p[1, :]) * (r[0, :] - q[0, :])) - ((q[0, :] - p[0, :]) * (r[1, :] - q[1, :]))
        orientation_mat = np.full(p.shape[1], 0)  # default Colinear orientation
        orientation_mat = np.where(val > 0, 1, orientation_mat)  # Clockwise orientation
        orientation_mat = np.where(val < 0, 2, orientation_mat)  # Counterclockwise orientation
        return orientation_mat
        # if (val > 0):
        #     # Clockwise orientation
        #     return 1
        # elif (val < 0):
        #     # Counterclockwise orientation
        #     return 2
        # else:
        #     # Colinear orientation
        #     return 0

    def onSegment(p, q, r):
        """Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
        """
        onsegment_mat = np.full(p.shape[1], True)  # default on Segment
        # if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        #     return True
        # return False
        onsegment_mat = np.logical_and(onsegment_mat, q[0, :] <= np.maximum(p[0, :], r[0, :]))
        onsegment_mat = np.logical_and(onsegment_mat, q[0, :] >= np.minimum(p[0, :], r[0, :]))
        onsegment_mat = np.logical_and(onsegment_mat, q[1, :] <= np.maximum(p[1, :], r[1, :]))
        onsegment_mat = np.logical_and(onsegment_mat, q[1, :] >= np.minimum(p[1, :], r[1, :]))
        return onsegment_mat

    # p1 and p2 to numpy array
    p1 = np.repeat([[p1_[0]], [p1_[1]]], p3.shape[1], axis=1)
    p2 = np.repeat([[p2_[0]], [p2_[1]]], p3.shape[1], axis=1)

    # Find the 4 orientations required for the general and special cases
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    intersection_mat = np.full(p3.shape[1], False)
    # General case
    # if ((o1 != o2) and (o3 != o4)):
    intersection_mat[np.logical_and(o1 != o2, o3 != o4)] = True
    # return True

    # Special Cases
    if including_online:
        intersection_mat[np.logical_and(o1 == 0, onSegment(p1, p3, p2))] = True
        intersection_mat[np.logical_and(o2 == 0, onSegment(p1, p4, p2))] = True
        intersection_mat[np.logical_and(o3 == 0, onSegment(p3, p1, p4))] = True
        intersection_mat[np.logical_and(o4 == 0, onSegment(p3, p2, p4))] = True
    else:
        # p1 , p2 and p3 are colinear and p3 lies on segment p1p2
        # if ((o1 == 0) and onSegment(p1, p3, p2)):
        #     return True
        intersection_mat[np.logical_and(o1 == 0, onSegment(p1, p3, p2))] = False
        # p1 , p2 and p4 are colinear and p4 lies on segment p1p2
        # if ((o2 == 0) and onSegment(p1, p4, p2)):
        #     return True
        intersection_mat[np.logical_and(o2 == 0, onSegment(p1, p4, p2))] = False
        # p3 , p4 and p1 are colinear and p1 lies on segment p3q4
        # if ((o3 == 0) and onSegment(p3, p1, p4)):
        #     return True
        intersection_mat[np.logical_and(o3 == 0, onSegment(p3, p1, p4))] = False
        # p3 , p4 and p2 are colinear and p2 lies on segment p3q4
        # if ((o4 == 0) and onSegment(p3, p2, p4)):
        #     return True
        intersection_mat[np.logical_and(o4 == 0, onSegment(p3, p2, p4))] = False

    return intersection_mat


def detect_intersection_segments(p1, p2, p3, p4):
    """Get intersction of segment.
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    :param p1: 1st line 1st points (x,y).
    :type p1: lists
    :param p2: 1st line 2nd points (x,y).
    :type p2: lists
    :param p3: 2nd line 1st points (x,y).
    :type p3: lists
    :param p4: 2nd line 2nd points (x,y).
    :type p4: lists
    :return: 2 given line segments intersect or not, returns true if the line segment 'p1p2' and 'p3p4' intersect.
    :rtype: numpy bool
    """
    def orientation(p, q, r):
        """ to find the orientation of an ordered triplet (p,q,r), 3 points (x,y)
        function returns the following values:
            0 : Colinear points
            1 : Clockwise points
            2 : Counterclockwise
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        """
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            # Clockwise orientation
            return 1
        elif (val < 0):
            # Counterclockwise orientation
            return 2
        else:
            # Colinear orientation
            return 0

    def onSegment(p, q, r):
        """Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
        """
        if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    # Find the 4 orientations required for the general and special cases
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
    # Special Cases
    # p1 , p2 and p3 are colinear and p3 lies on segment p1p2
    if ((o1 == 0) and onSegment(p1, p3, p2)):
        return True
    # p1 , p2 and p4 are colinear and p4 lies on segment p1p2
    if ((o2 == 0) and onSegment(p1, p4, p2)):
        return True
    # p3 , p4 and p1 are colinear and p1 lies on segment p3q4
    if ((o3 == 0) and onSegment(p3, p1, p4)):
        return True
    # p3 , p4 and p2 are colinear and p2 lies on segment p3q4
    if ((o4 == 0) and onSegment(p3, p2, p4)):
        return True

    # If none of the cases
    return False


def find_intersection_line(p1,  p2,  p3,  p4):
    """
    1st line p1->p2, 2nd line p3->p4.

    :param p1: 1st line 1st points (x,y).
    :type p1: lists
    :param p2: 1st line 2nd points (x,y).
    :type p2: lists
    :param p3: 2nd line 1st point. [2, points_number]
    :type p3: numpy
    :param p4: 2nd line 2nd point. [2, points_number]
    :type p4: numpy
    """
    # the segments
    dx12 = p2[0] - p1[0]
    dy12 = p2[1] - p1[1]
    dx34 = p4[0, :] - p3[0, :]
    dy34 = p4[1, :] - p3[1, :]
    denominator = (dy12 * dx34 - dx12 * dy34)
    nonzero_index = (denominator != 0)
    t1 = ((p1[0] - p3[0, :][nonzero_index]) * dy34 + (p3[1, :][nonzero_index] - p1[1]) * dx34) / denominator[nonzero_index]
    # point of intersection.
    intersection = np.full_like(p3, np.nan)
    intersection[:, nonzero_index] = [p1[0] + dx12 * t1, p1[1] + dy12 * t1]
    return intersection


def find_intersection(p1,  p2,  p3,  p4):
    """Find the point of intersection between two line.
    
    The two lines are p1 --> p2 and p3 --> p4.
    Reference:http://csharphelper.com/blog/2020/12/enlarge-a-polygon-that-has-colinear-vertices-in-c/

    :param p1: line 1's start point, 2D point (x,y)
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


def is_clockwise(point_list):
    """Check whether the list is clockwise. 

    :param point_list: the point lists
    :type point_list: numpy 
    :return: yes if the points are clockwise, otherwise is no
    :rtype: bool
    """
    sum = 0
    for i in range(len(point_list)):
        cur = point_list[i]
        next = point_list[(i+1) % len(point_list)]
        sum += (next[0] - cur[0]) * (next[1] + cur[1])
    return sum > 0


def enlarge_polygon(old_points, offset):
    """Return points representing an enlarged polygon.

    Reference: http://csharphelper.com/blog/2016/01/enlarge-a-polygon-in-c/
    
    :param old_points: the polygon vertexes, and the points should be in clock wise
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
        if not is_clockwise(old_points):
            log.error("the points list is not clockwise.")

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
