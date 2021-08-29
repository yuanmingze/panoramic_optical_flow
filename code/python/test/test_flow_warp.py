import configuration

from utility import flow_warp
from utility import flow_vis
from utility import spherical_coordinates as sc
from utility import image_io
from utility import mocker_data_generator as MDG

from scipy.spatial.transform import Rotation as R


def test_flow2rotation():
    erp_height = 100
    erp_width = erp_height * 2

    # 0) create flow
    rotation_matrix = R.from_euler("zyx", [5, 10, 0], degrees=True).as_matrix()
    print(R.from_matrix(rotation_matrix).as_euler("zyx", degrees=True))
    erp_motion_vector, _ = sc.rotation2erp_motion_vector((erp_height, erp_width),     rotation_matrix=rotation_matrix)
    flow_color = flow_vis.flow_to_color(erp_motion_vector)
    # image_io.image_show(flow_color)

    # 1) convert to rotation
    rotation_mat = flow_warp.flow2rotation_3d(erp_motion_vector)

    # 2) output result with euler angles
    rotation_euler = R.from_matrix(rotation_mat).as_euler('zyx', degrees=True)
    print(rotation_euler)

    assert((rotation_mat == rotation_matrix).any())

def test_global_rotation_warping():
    erp_height = 100
    erp_width = erp_height * 2

    # 0) create flow & image
    rotation_matrix = R.from_euler("zyx", [20, 30, 0], degrees=True).as_matrix()
    print(R.from_matrix(rotation_matrix).as_euler("zyx", degrees=True))
    erp_motion_vector, _ = sc.rotation2erp_motion_vector((erp_height, erp_width),     rotation_matrix=rotation_matrix)
    flow_color = flow_vis.flow_to_color(erp_motion_vector)
    image_io.image_show(flow_color)

    image_data = MDG.get_erp_image()

    # 1) global rotation warping
    image_data_rotated, rotation_mat = flow_warp.global_rotation_warping(image_data, erp_motion_vector)

    # 2) output result with euler angles
    # rotation_euler = R.from_matrix(rotation_mat).as_euler('zyx', degrees=True)
    # print(rotation_euler)
    image_io.image_show(image_data_rotated)

if __name__ == "__main__":
    # test_flow2rotation()
    test_global_rotation_warping()
