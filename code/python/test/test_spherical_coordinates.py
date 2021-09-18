
import configuration as config
import spherical_coordinates

from utility import spherical_coordinates as sc
from utility import image_io
from utility import flow_warp
from utility import flow_vis

import numpy as np
from scipy.spatial.transform import Rotation as R, rotation
import os
import math

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False

import unittest

class TestSphericalCoordinates(unittest.TestCase):

    def setUp(self) -> None:
        self.erp_image_height = 512
        self.erp_image_width = self.erp_image_height * 2
        self.erp_points = np.zeros([2, 9], np.float32)
        self.erp_points[:, 0] = [0, 0]
        self.erp_points[:, 1] = [self.erp_image_width - 1, 0]
        self.erp_points[:, 2] = [0, self.erp_image_height-1]
        self.erp_points[:, 3] = [self.erp_image_width - 1, self.erp_image_height-1]
        # center corners
        self.erp_points[:, 4] = [(self.erp_image_width - 1) / 2.0, (self.erp_image_height - 1) / 2.0]
        # four corners
        self.erp_points[:, 5] = [-0.5, -0.5]
        self.erp_points[:, 6] = [self.erp_image_width - 1 + 0.5 , -0.5]
        self.erp_points[:, 7] = [-0.5, self.erp_image_height - 1 + 0.5]
        self.erp_points[:, 8] = [self.erp_image_width - 1 + 0.5 , self.erp_image_height - 1 + 0.5]

        self.sph_points = np.zeros([2, 9], np.float32)
        self.sph_points[:, 0] = [-np.pi + (np.pi * 2 / self.erp_image_width) / 2.0, np.pi / 2.0 - (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 1] = [np.pi - (np.pi * 2 / self.erp_image_width) / 2.0, np.pi / 2.0 - (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 2] = [-np.pi + (np.pi * 2 / self.erp_image_width) / 2.0, -np.pi / 2.0 + (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 3] = [np.pi - (np.pi * 2 / self.erp_image_width) / 2.0, -np.pi / 2.0 + (np.pi / self.erp_image_height) / 2.0]
        # center corners
        self.sph_points[:, 4] = [0, 0]
        # four corners
        self.sph_points[:, 5] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 6] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 7] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 8] = [-np.pi, np.pi/2.0]

        return super().setUp()

    def test_erp2sph(self):
        sph_points_from_erp = sc.erp2sph(self.erp_points, self.erp_image_height, True)
        # print("--:\n{}".format(sph_points_from_erp.T))
        # print("GT:\n{}".format(self.sph_points.T))
        self.assertTrue(np.allclose(sph_points_from_erp, self.sph_points))

    def test_sph2erp(self):
        erp_points_from_sph = sc.sph2erp_0(self.sph_points, self.erp_image_height, True)
        temp = sc.erp_pixel_modulo_0(self.erp_points, self.erp_image_height)
        # print("--:\n{}".format(erp_points_from_sph.T))
        # print("GT:\n{}".format(temp.T))
        self.assertTrue(np.allclose(erp_points_from_sph, temp, atol=1e-4, rtol=1e-4))


def test_get_angle_ofcolor():
    """ Test the point from the spherical surface."""
    points_number = 1000
    r = np.pi * 0.15

    # pointA = np.ones((points_number, 2), dtype=np.float64) * np.pi * 0.25
    # center point
    point_center_u = 0.0
    point_center_v = 0.0
    # point_center_u = -np.pi * 0.2
    # point_center_v = -point_center_u * 0.5
    # point_center_u = 0
    # # TODO Fix bug!!
    # point_center_v = np.pi * 0.5
    # point_v[:, 1]  =  np.pi * 0.5

    point_center = np.zeros((points_number, 2), dtype=np.float64)
    point_center[:, 0] = point_center_u
    point_center[:, 1] = point_center_v

    # points of u
    point_u = np.zeros((points_number, 2), dtype=np.float64)  # theta, phi
    # point_u[:, 0] = 0.125 * np.pi  # theta
    point_u[:, 1] = point_center_v

    # points of v
    point_v = np.zeros((points_number, 2), dtype=np.float64)
    point_v[:, 0] = point_center_u

    for idx in range(0, points_number):
        x = np.cos(2*np.pi / points_number * idx) * r
        y = -np.sin(2 * np.pi / points_number * idx) * r
        point_u[idx, 0] = x + point_center_u
        point_v[idx, 1] = y + point_center_v

    # Modulo the theta and phi

    point_u[:, 0], point_v[idx, 1] = spherical_coordinates.erp_sph_modulo(point_u[:, 0], point_v[idx, 1])

    import matplotlib.pyplot as plt
    # plt.plot(pointC[:, 0], pointC[:, 1])
    # plt.axis('tight')

    # print(math.degrees((sc.get_angle_from_length(math.radians(76.4111), math.radians(58.31),  math.radians(105.74295)))))  # 118.50778
    # angle_sph_result = sc.get_angle_uv(point_center[:, 0], point_center[:, 1], point_u[:, 0], point_u[:, 1], point_v[:, 0], point_v[:, 1])

    # from utility import polygon
    # angle_cartesian_result = polygon.get_angle_from_vector(pointB- pointA, pointC-pointA)
    angle_cartesian_result = spherical_coordinates.get_angle_sph_ofcolor(point_center, point_u, point_v)
    # print(angle_cartesian_result)

    plt_x_index = np.linspace(0, points_number, endpoint=False, num=points_number)
    # plt.plot(plt_x_index, angle_sph_result)
    scatter_angle = plt.scatter(plt_x_index, angle_cartesian_result, c=plt_x_index, label='angle')
    # plot_u = plt.scatter(plt_x_index, point_u[:, 0], c="r", marker="o", label='u')
    # plot_v = plt.scatter(plt_x_index, point_v[:, 1], c="g", marker="s", label='v')
    # plt.plot(point_u[:,0], point_v[:,1])
    # plt.scatter(point_u[:,0], point_v[:,1], c=plt_x_index)
    plt.legend(handles=[scatter_angle])#, plot_u,plot_v])
    # plt.colorbar()
    plt.show()


def test_get_angle_uv_random():
    # 1) random sample
    rng = np.random.default_rng(12345)
    points_number = 10
    points_A_sph_list = np.zeros((points_number, 2), dtype=np.float64)
    points_A_sph_list[:, 0] = rng.uniform(low=-np.pi, high=np.pi, size=points_number)  # theta
    points_A_sph_list[:, 1] = rng.uniform(low=-np.pi * 0.5, high=np.pi * 0.5, size=points_number)

    points_B_sph_list = np.zeros((points_number, 2), dtype=np.float64)
    points_B_sph_list[:, 0] = rng.uniform(low=-np.pi, high=np.pi, size=points_number)
    points_B_sph_list[:, 1] = rng.uniform(low=-np.pi * 0.5, high=np.pi * 0.5, size=points_number)

    points_C_sph_list = np.zeros((points_number, 2), dtype=np.float64)
    points_C_sph_list[:, 0] = rng.uniform(low=-np.pi, high=np.pi, size=points_number)
    points_C_sph_list[:, 1] = rng.uniform(low=-np.pi * 0.5, high=np.pi * 0.5, size=points_number)

    # 2) get length from spherical_geometry
    import spherical_geometry.great_circle_arc
    # point_pair_np_3d = sc.erp2sph(point)
    points_A_car_list = sc.sph2car(points_A_sph_list[:, 0], points_A_sph_list[:, 1])
    points_B_car_list = sc.sph2car(points_B_sph_list[:, 0], points_B_sph_list[:, 1])
    points_C_cat_list = sc.sph2car(points_C_sph_list[:, 0], points_C_sph_list[:, 1])
    angle_sg_degree = spherical_geometry.great_circle_arc.angle(points_B_car_list.T, points_A_car_list.T, points_C_cat_list.T)
    angle_sg_radian = np.radians(angle_sg_degree)
    angle_sg_radian[angle_sg_radian > np.pi] = np.pi * 2.0 - angle_sg_radian[angle_sg_radian > np.pi]

    # 3) get length from myself code
    # length_our_radian = sc.great_circle_distance_uv(points_A_sph_list[:, 0], points_A_sph_list[:, 1], points_B_sph_list[:, 0], points_B_sph_list[:, 1])
    angle_our_radian = sc.get_angle_sph(points_A_sph_list, points_B_sph_list, points_C_sph_list)
    result = np.isclose(angle_sg_radian, angle_our_radian, atol=1e-08)
    assert(result.all())


def test_get_angle_uv():
    # 0) from http://128.192.17.191/EMAT6680Fa08/Broderick/unit/day2.html
    # print(sc.get_angle_from_length([0.136 * np.pi], [0.303 * np.pi],  [0.226 * np.pi]) / np.pi)  # 0.261
    # print(sc.get_angle_from_length([0.226 * np.pi], [0.303 * np.pi],  [0.136 * np.pi]) / np.pi)  # 0.153
    # print(sc.get_angle_from_length([0.226 * np.pi], [0.136 * np.pi],  [0.303 * np.pi]) / np.pi)  # 0.633


    # length_AB_arr = np.array([0.136 * np.pi, 0.303 * np.pi, 0.226 * np.pi, 0.226 * np.pi])
    # length_AC_arr = np.array([0.303 * np.pi, 0.136 * np.pi, 0.303 * np.pi, 0.136 * np.pi])
    # length_BC_arr = np.array([0.226 * np.pi, 0.226 * np.pi, 0.136 * np.pi, 0.303 * np.pi])
    # pointA_list = np.array([], dtype=np.float64)
    # pointB_list = np.array([], dtype=np.float64)
    # pointC_list = np.array([], dtype=np.float64)
    # angle_A_arr = sc.get_angle_sph(pointA_list, pointB_list, pointC_list)
    # print(angle_A_arr)

    points_number = 100
    r = np.pi * 0.15
    # center point
    point_center = np.zeros((points_number, 2), dtype=np.float64)
    # points of u
    point_B = np.zeros((points_number, 2), dtype=np.float64)  # theta, phi
    # points of v
    point_C = np.zeros((points_number, 2), dtype=np.float64)
    point_C[:, 0] = 0.2

    for idx in range(0, points_number):
        x = np.cos(2*np.pi / points_number * idx) * r
        y = -np.sin(2 * np.pi / points_number * idx) * r
        point_B[idx, 0] = x  # theta
        point_B[idx, 1] = y  # phi

    import matplotlib.pyplot as plt
    angle_cartesian_result = spherical_coordinates.get_angle_sph(point_center,  point_C, point_B)
    plt_x_index = np.linspace(0, points_number, endpoint=False, num=points_number)
    # plt.plot(plt_x_index, angle_sph_result)
    plt.scatter(plt_x_index, angle_cartesian_result, c=plt_x_index)
    # plt.plot(plt_x_index, point_u)
    # plt.plot(plt_x_index, point_v)
    # plt.plot(point_u[:,0], point_v[:,1])
    plt.scatter(point_B[:, 0], point_B[:, 1], c=plt_x_index)
    plt.colorbar()
    plt.show()


def test_great_circle_distance_uv_random():
    # the gt form module: https://pypi.org/project/spherical-geometry/

    # 1) random sample
    rng = np.random.default_rng(12345)
    points_number = 100
    points_src_sph_list = np.zeros((points_number, 2), dtype=np.float64)
    points_src_sph_list[:, 0] = rng.uniform(low=-np.pi, high=np.pi, size=points_number)  # theta
    points_src_sph_list[:, 1] = rng.uniform(low=-np.pi * 0.5, high=np.pi * 0.5, size=points_number)

    points_tar_sph_list = np.zeros((points_number, 2), dtype=np.float64)
    points_tar_sph_list[:, 0] = rng.uniform(low=-np.pi, high=np.pi, size=points_number)
    points_tar_sph_list[:, 1] = rng.uniform(low=-np.pi * 0.5, high=np.pi * 0.5, size=points_number)

    # 2) get length from spherical_geometry
    import spherical_geometry.great_circle_arc
    # point_pair_np_3d = sc.erp2sph(point)
    points_src_car_list = sc.sph2car(points_src_sph_list[:, 0], points_src_sph_list[:, 1])
    points_tar_car_list = sc.sph2car(points_tar_sph_list[:, 0], points_tar_sph_list[:, 1])
    length_sg_degree = spherical_geometry.great_circle_arc.length(points_src_car_list.T, points_tar_car_list.T)
    length_sg_radian = np.radians(length_sg_degree)

    # 3) get length from myself code
    length_our_radian = sc.great_circle_distance_uv(points_src_sph_list[:, 0], points_src_sph_list[:, 1], points_tar_sph_list[:, 0], points_tar_sph_list[:, 1])

    result = np.isclose(length_sg_radian, length_our_radian, atol=1e-08)
    assert(result.all())


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
    point_pair_np = np.array(point_pair)
    # point_pair.append([[0, -np.pi / 2.0 + 0.001], [0, np.pi / 2.0]])

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

    if np.allclose(result, result_comput):
        log.info("Pass")
    else:
        for index in range(len(point_pair)):
            print("----{}-----{}".format(index, point_pair[index]))
            print("error:    {}".format(np.sqrt(np.abs(result_comput[index] - result[index]))))
            print("GT:       {}".format(result[index]))
            print("Computed: {}".format(result_comput[index]))
            if not np.isclose(result[index], result_comput[index]):
                log.error(" ")


def test_rotation2erp_motion_vector(erp_src_image_filepath):
    """" Test use rotation information to create optical flow."""
    src_image_data = image_io.image_read(erp_src_image_filepath)
    image_size = src_image_data.shape[0:2]
    rotation_theta = np.radians(10.0)
    rotation_phi = np.radians(00.0)
    # rotation_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi)
    # erp_motion_vector= sc.rotation2erp_motion_vector(image_size, rotation_mat)
    # flow_color = flow_vis.flow_to_color(erp_motion_vector)
    # image_io.image_show(flow_color)

    # image_io.image_save(flow_color, erp_src_image_filepath + "_flow.jpg")
    # flow_color = flow_vis.flow_value_to_color(erp_motion_vector)
    # erp_motion_vector = sc.rotation2erp_motion_vector(image_size, rotation_theta, rotation_phi,  wraparound=True)
    # flow_color = flow_vis.flow_value_to_color(erp_motion_vector)

    # with the rotation matrix
    # rotaion_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi, False)
    # rot_alpha = 10.0
    # rot_beta = 20.0
    # rot_gema = 15.0
    rotation_mat = R.from_euler("xyz", [10.0, 15.0, 20], degrees=True)
    erp_motion_vector= sc.rotation2erp_motion_vector(image_size, rotation_matrix=rotation_mat)
    erp_motion_vector = flow_vis.flow_to_color(erp_motion_vector)
    image_io.image_show(erp_motion_vector)


def test_rotate_erp_array_mat(erp_src_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)
    image_io.image_show(src_image_data)

    rotation_mat = R.from_euler("xyz", [10, 15,20], degrees = True).as_matrix()
    src_image_data_rotated = sc.rotate_erp_array(src_image_data, rotation_mat=rotation_mat)
    image_io.image_show(src_image_data_rotated)

    src_image_data_restore  = sc.rotate_erp_array(src_image_data_rotated, rotation_mat=rotation_mat.T)
    image_io.image_show(src_image_data_restore)
 

def test_rotate_erp_array(erp_src_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)

    # double side rotation
    # rotation_theta = np.radians(50.0)
    # rotation_phi = np.radians(30.0)
    # rotation_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi, False)
    rotation_mat = R.from_euler("xyz", [45,0,0], degrees= True).as_matrix()
    tar_image_data_rot = sc.rotate_erp_array(src_image_data, rotation_mat)
    image_io.image_show(tar_image_data_rot)

    # tar_image_data_rot_skylib = sc.rotate_erp_array_skylib(src_image_data, rotation_mat)
    # image_io.image_show(tar_image_data_rot_skylib)

    # image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_rot_0.jpg")
    tar_image_data_rot = sc.rotate_erp_array(tar_image_data_rot, rotation_mat=rotation_mat.T)
    image_io.image_show(tar_image_data_rot)
    # image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_rot_1.jpg")

    # multi wrap around
    # rotation_theta = np.radians(90.0)
    # rotation_phi = np.radians(0.0)
    rotation_theta = np.radians(0.0)
    rotation_phi = np.radians(45.0)
    tar_image_data_rot = src_image_data
    for index in range(0, 4):
        rotate_mat = sc.rot_sph2mat(rotation_theta, rotation_phi)
        tar_image_data_rot = sc.rotate_erp_array(tar_image_data_rot, rotate_mat)
    # image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_rot_2.jpg")
    image_io.image_show(tar_image_data_rot)


def test_rotate_array_coord(erp_src_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)
    image_size = src_image_data.shape[0:2]
    rotation_theta = np.radians(0.0)
    rotation_phi = np.radians(30.0)

    #  rotate
    rotate_mat = sc.rot_sph2mat(rotation_theta, rotation_phi)
    tar_image_data_rot = sc.rotate_erp_array(src_image_data, rotate_mat)
    image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_rot.jpg")

    # self rotate
    rotation_max = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi)
    erp_motion_vector = sc.rotation2erp_motion_vector(image_size, rotation_max)
    # erp_motion_vector = np.moveaxis(erp_motion_vector, 0, -1)

    erp_motion_vector_vis = flow_vis.flow_to_color(erp_motion_vector)
    image_io.image_save(erp_motion_vector_vis, erp_src_image_filepath + "_flow_vis.jpg")

    # src_image_data_rot = flow_warp.warp_forward(tar_image_data_rot, erp_motion_vector)
    # image_io.image_save(src_image_data_rot, erp_src_image_filepath + "_warp_forward_rot.jpg")

    tar_image_data_rot = flow_warp.warp_backward(tar_image_data_rot, erp_motion_vector)
    image_io.image_save(tar_image_data_rot, erp_src_image_filepath + "_warp_backward_rot.jpg")


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')

    args = parser.parse_args()

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    
    test_list = []
    test_list.append(args.task)

    if 0 in test_list:
        test_great_circle_distance_uv()
    if 1 in test_list:
        test_rotate_array_coord(erp_src_image_filepath)
    if 2 in test_list:
        test_rotate_erp_array(erp_src_image_filepath)
    if 3 in test_list:
        test_rotation2erp_motion_vector(erp_src_image_filepath)
    if 4 in test_list:
        test_get_angle_uv()
    if 5 in test_list:
        unittest.main()
    if 6 in test_list:
        test_rotate_erp_array_mat(erp_src_image_filepath)
    if 7 in test_list:
        test_get_angle_ofcolor()
    if 8 in test_list:
        test_great_circle_distance_uv_random()
    if 9 in test_list:
        test_get_angle_uv_random()
