
import configuration as config

from utility import spherical_coordinates as sc
from utility import image_io
from utility import flow_warp
from utility import flow_vis

from utility.logger import Logger

import numpy as np
import os
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


log = Logger(__name__)
log.logger.propagate = False


def test_great_circle_distance_uv():
    """"test great_circle_distance
    """
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
    point_pair.append([[0.0, -np.pi / 4.0], [0.0, np.pi / 4.0]])
    point_pair.append([[-np.pi / 4.0, np.pi / 4.0], [np.pi / 4.0, np.pi / 4.0]])
    point_pair.append([[-np.pi / 6.0, -np.pi / 8.0], [np.pi / 4.0, 0.0]])

    result = [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2, 1.0471975511965979, 1.32933932]

    points_1_u = np.zeros(len(point_pair))
    points_1_v = np.zeros(len(point_pair))
    points_2_u = np.zeros(len(point_pair))
    points_2_v = np.zeros(len(point_pair))
    for index in range(len(point_pair)):
        # term = point_pair[index]
        points_1_u[index] = point_pair[index][0][0]
        points_1_v[index] = point_pair[index][0][1]

        points_2_u[index] = point_pair[index][1][0]
        points_2_v[index] = point_pair[index][1][1]

    points_1_u = points_1_u[np.newaxis, ...]
    points_1_v = points_1_v[np.newaxis, ...]
    points_2_u = points_2_u[np.newaxis, ...]
    points_2_v = points_2_v[np.newaxis, ...]

    result_comput = sc.great_circle_distance_uv(points_1_u, points_1_v, points_2_u, points_2_v, radius=1)
    result_comput = result_comput[0]

    for index in range(len(point_pair)):
        print("----{}-----{}".format(index, point_pair[index]))
        print("error:    {}".format(np.sqrt(np.abs(result_comput[index] - result[index]))))
        print("GT:       {}".format(result[index]))
        print("Computed: {}".format(result_comput[index]))
        if not np.isclose(result[index], result_comput[index]):
            log.error(" ")


def test_rotate_array_coord(erp_src_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)
    image_size = [960, 480]
    rotation_theta = np.radians(0.0)
    rotation_phi = np.radians(30.0)

    # # envmap rotate
    # tar_image_data_rot = sc.rotate_array(src_image_data, rotation_theta, rotation_phi)
    # image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_rot.jpg")

    # self rotate
    erp_motion_vector = sc.rotate_erp_motion_vector(image_size, -rotation_theta, -rotation_phi)
    erp_motion_vector = np.moveaxis(erp_motion_vector, 0, -1)
    src_image_data_rot = flow_warp.warp_forward(src_image_data, erp_motion_vector)
    erp_motion_vector_vis = flow_vis.flow_to_color(erp_motion_vector)
    image_io.image_save(erp_motion_vector_vis, erp_src_image_filepath + "_flow_vis.jpg")
    image_io.image_save(src_image_data_rot, erp_src_image_filepath + "_warp_rot.jpg")


if __name__ == "__main__":
    # test_great_circle_distance_uv()

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    test_rotate_array_coord(erp_src_image_filepath)
