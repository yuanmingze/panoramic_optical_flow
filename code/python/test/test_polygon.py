import os
import configuration as config
from utility import polygon
import numpy as np
import matplotlib.pyplot as plt


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

    intersecion_point = polygon.enlarge_polygon(old_points, 0.3)
    x_coordinates = []
    y_coordinates = []
    for term in intersecion_point:
        x_coordinates.append(term[0])
        y_coordinates.append(term[1])
    x_coordinates.append(intersecion_point[0][0])
    y_coordinates.append(intersecion_point[0][1])

    # import ipdb; ipdb.set_trace()

    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinates, y_coordinates)
    plt.show()

if __name__ == "__main__":
    # test_find_intersection()
    test_enlarge_polygon()
