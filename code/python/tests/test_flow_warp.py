import configuration

import numpy as np
from scipy.spatial.transform import Rotation as R

from utility import spherical_coordinates
from utility import flow_warp
from utility import flow_vis
from utility import spherical_coordinates as sc
from utility import image_io
from utility import flow_io
from utility import mocker_data_generator as MDG
from utility import flow_postproc


def test_warp_forward():
    data_root = configuration.TEST_data_root_dir + "replica_360/apartment_0/"
    rgb_filepath = data_root + "0001_rgb.jpg"
    rgb_data = image_io.image_read(rgb_filepath)
    of_filepath = data_root + "0001_opticalflow_forward.flo"
    of_data = flow_io.read_flow_flo(of_filepath)

    rgb_warp_filepath = data_root + "0001_rgb_warp.jpg"
    rgb_warp = flow_warp.warp_forward(rgb_data, of_data, wrap_around=True)
    print("Output warped image to {}".format(rgb_warp_filepath))
    image_io.image_save(rgb_warp, rgb_warp_filepath)


def test_flow2rotation():
    erp_height = 100
    erp_width = erp_height * 2

    # 0) create flow
    rotation_matrix = spherical_coordinates.rot_sph2mat(10, 5, True)
    print("The ground truch rotation is {}".format(spherical_coordinates.rot_mat2sph(rotation_matrix)))
    erp_motion_vector = sc.rotation2erp_motion_vector((erp_height, erp_width), rotation_matrix=rotation_matrix)
    flow_color = flow_vis.flow_to_color(erp_motion_vector)
    image_io.image_show(flow_color)

    # 1) convert to rotation
    estimation_method = "2D"
    if estimation_method == "3D":
        rotation_mat = flow_warp.flow2rotation_3d(erp_motion_vector, mask_method="center")
    elif estimation_method == "2D":
        theta, phi = flow_warp.flow2rotation_2d(erp_motion_vector, False)
        rotation_mat = sc.rot_sph2mat(theta, phi, False)

    # 2) output result with euler angles
    rotation_euler = spherical_coordinates.rot_mat2sph(rotation_mat)
    print("The estimated rotation is {}".format(rotation_euler))


def test_flow2rotation_2d(erp_src_image_filepath, erp_tar_image_filepath):
    src_image_data = image_io.image_read(erp_src_image_filepath)
    tar_image_data = image_io.image_read(erp_tar_image_filepath)
    # flow_array = flow_estimate.of_methdod_DIS(src_image_data, tar_image_data)

    # theta, phi
    rotation_list = [(-20, 10.0), (30.0, -10.0), (30.0, 15.0), (-30.0, 15.0), (-25.0, -13.0)]
    for rotation in rotation_list:
        rotation_theta = np.radians(rotation[0])
        rotation_phi = np.radians(rotation[1])
        rotation_mat = spherical_coordinates.rot_sph2mat(rotation_theta, rotation_phi)
        flow_array = spherical_coordinates.rotation2erp_motion_vector(src_image_data.shape[0:2], rotation_mat)
        # flow_vis.flow_value_to_color(flow_array)
        theta_delta, phi_delta = flow_warp.flow2rotation_2d(flow_array, False)
        print("original rotation: {},{}, result is: {}, {}".
              format(np.degrees(rotation_theta), np.degrees(rotation_phi), np.degrees(theta_delta), np.degrees(phi_delta)))


def test_global_rotation_warping():
    erp_height = 100
    erp_width = erp_height * 2

    # 0) create flow & image
    # delta_theta = 30
    # delta_phi = 20
    # rotation_matrix = spherical_coordinates.rot_sph2mat(delta_theta, delta_phi, True)
    # print("{}".format(spherical_coordinates.rot_mat2sph(rotation_matrix)))

    rotation_matrix = R.from_euler("xyz", [10, 20, 0], degrees=True).as_matrix()
    erp_motion_vector = sc.rotation2erp_motion_vector((erp_height, erp_width), rotation_matrix=rotation_matrix)
    # flow_color = flow_vis.flow_to_color(erp_motion_vector)
    # image_io.image_show(flow_color)

    image_data = MDG.image_square()
    image_io.image_show(image_data)

    # 1) global rotation warping
    estimation_method = "2D"
    forward_warp = False
    if estimation_method == "3D":
        image_data_rotated, rotation_mat = flow_warp.global_rotation_warping(image_data, erp_motion_vector, forward_warp=forward_warp, rotation_type="3D")
    elif estimation_method == "2D":
        image_data_rotated, rotation_mat = flow_warp.global_rotation_warping(image_data, erp_motion_vector, forward_warp=forward_warp, rotation_type="2D")

    # 2) output result with euler angles
    # rotation_euler = R.from_matrix(rotation_mat).as_euler('XYZ', degrees=True) #
    # print(rotation_euler)
    # print("{}".format(spherical_coordinates.rot_mat2sph(rotation_mat)))
    image_io.image_show(image_data_rotated)

    image_data_restore = sc.rotate_erp_array(image_data_rotated, rotation_mat=rotation_mat.T)
    image_io.image_show(image_data_restore)


def test_global_rotation_warping_0(erp_src_image_path, erp_flow_path):
    """Test the get_rotation function.
    """
    erp_image = image_io.image_read(erp_src_image_path)
    erp_flow = flow_io.flow_read(erp_flow_path)

    erp_flow = flow_postproc.erp_of_wraparound(erp_flow)

    # get the rotation
    erp_image_rotated = flow_warp.global_rotation_warping(erp_image, erp_flow)

    # rotate the src image
    rotated_image_path = erp_src_image_path + "_rotated.jpg"
    print("output rotated image: {}".format(rotated_image_path))
    image_io.image_save(erp_image_rotated, rotated_image_path)


if __name__ == "__main__":
    test_list = [2]
    if 0 in test_list:
        test_flow2rotation()
    if 1 in test_list:
        test_global_rotation_warping()
    if 2 in test_list:
        test_warp_forward()
