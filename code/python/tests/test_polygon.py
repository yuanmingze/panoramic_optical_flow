import configuration as config

import os
import numpy as np
import matplotlib.pyplot as plt

import polygon


def test_detect_intersection_segments_array():
    p1 = (-5, -2)
    p2 = (0, 1)
    p3 = np.array([[1, 1], [10, 0], [0, 0], [20, 21], [0, 1], [0, -1]]).T
    p4 = np.array([[10, 10], [10, 0], [10, 2], [30, 31], [-1, -3], [-4, 3]]).T
    print(polygon.detect_intersection_segments_array(p1, p2, p3, p4))


def test_detect_intersection_segments():
    p1 = (-5, -2)
    p2 = (0, 1)
    ####
    p3 = (1, 1)
    p4 = (10, 10)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))
    p3 = (10, 0)
    p4 = (8, 0)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))
    p3 = (0, 0)
    p4 = (10, 2)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))
    p3 = (20, 21)
    p4 = (30, 31)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))
    p3 = (0, 1)
    p4 = (-1, -3)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))
    p3 = (0, -1)
    p4 = (-4, 3)
    print(polygon.detect_intersection_segments(p1, p2, p3, p4))


def test_find_intersection():
    """
    """
    p1 = [0, 2]
    p2 = [0, 1]
    p3 = [1, 0]
    p4 = [2, 0]
    intersecion_point = polygon.find_intersection(p1, p2, p3, p4)
    assert(intersecion_point == [0.0, 0.0])
    print(intersecion_point)

    p1 = [1, 1]
    p2 = [2, 2]
    p3 = [1, -1]
    p4 = [2, -2]
    intersecion_point = polygon.find_intersection(p1, p2, p3, p4)
    assert(intersecion_point == [0.0, 0.0])
    print(intersecion_point)

    p1 = [0, 0]
    p2 = [2, 0]
    p3 = [0, 2]
    p4 = [1, 1]
    intersecion_point = polygon.find_intersection(p1, p2, p3, p4)
    assert(intersecion_point == [2.0, 0.0])
    print(intersecion_point)


def test_find_intersection_array():
    """
    """
    p1 = [0, 2]
    p2 = [0, 1]
    p3 = np.array([[1, 0], [1, -1], [0, 2]], np.float32)
    p4 = np.array([[2, 0], [2, -2], [1, 1]], np.float32)
    intersecion_point = polygon.find_intersection_array(p1, p2, p3.T, p4.T)
    print(intersecion_point)
    assert((intersecion_point == [[0., 0., 0.], [0., 0., 2.]]).all())


def test_enlarge_polygon():
    """[summary]
    """
    x_coordinates = []
    y_coordinates = []

    old_points = [[-1, -1], [0, 1], [1, -1]]

    for term in old_points:
        x_coordinates.append(term[0])
        y_coordinates.append(term[1])
    x_coordinates.append(old_points[0][0])
    y_coordinates.append(old_points[0][1])
    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinates, y_coordinates)

    intersecion_point = polygon.enlarge_polygon(old_points, 0.5)
    x_coordinates = []
    y_coordinates = []
    for term in intersecion_point:
        x_coordinates.append(term[0])
        y_coordinates.append(term[1])
    x_coordinates.append(intersecion_point[0][0])
    y_coordinates.append(intersecion_point[0][1])

    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinates, y_coordinates)
    plt.show()


def test_is_clockwise():
    point_list = [[-1, 0], [0, 1], [1, 0]]
    assert(polygon.is_clockwise(point_list))

    point_list = [[0, 3], [3, 1], [1, -3]]
    assert(polygon.is_clockwise(point_list))

    point_list = [[0, 3], [1, -3], [3, 1]]
    assert(polygon.is_clockwise(point_list) == False)

    point_list = [[0, 1], [-1, 0], [1, 0]]
    assert(polygon.is_clockwise(point_list) == False)

def test_get_angle_from_vector():
    points_number = 4
    vector1 = np.zeros((points_number, 3), dtype= np.float64)
    vector2 = np.zeros((points_number, 3), dtype= np.float64)

    # set the data
    vector1[:, 0] = 0
    vector1[:, 1] = 1
    vector1[:, 2] = 0
    vector2[0, :] = [1, 1, 0]
    vector2[1, :] = [-1, 1, 0]
    vector2[2, :] = [-1, -1, 0]
    vector2[3, :] = [1, -1, 0]

    # get the euler angle
    angle_list = polygon.get_angle_from_vector(vector1, vector2)
    angle_list = np.degrees(angle_list)
    print("Test 3D: The angle is \n{}".format(angle_list))

    vector1 = np.zeros((points_number, 2), dtype= np.float64)
    vector2 = np.zeros((points_number, 2), dtype= np.float64)

    # set the data
    vector1[:, 0] = 0
    vector1[:, 1] = 1
    vector2[0, :] = [1, 1]
    vector2[1, :] = [-1, 1]
    vector2[2, :] = [-1, -1]
    vector2[3, :] = [1, -1]
    # get the euler angle
    angle_list = polygon.get_angle_from_vector(vector1, vector2)
    angle_list = np.degrees(angle_list)
    print("Test 2D: The angle is \n{}".format(angle_list))


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')

    args = parser.parse_args()

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    
    test_list = []
    test_list.append(args.task)

    if 0 in test_list:
        test_get_angle_from_vector()
    if 1 in test_list:
        # test_find_intersection()
        # test_find_intersection_array()
        # test_enlarge_polygon()
        # test_is_clockwise()
        test_detect_intersection_segments()
        test_detect_intersection_segments_array()
