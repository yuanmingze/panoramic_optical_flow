import configuration as config

from scipy.spatial.transform import Rotation as R
import numpy as np

from utility import pointcloud_utils

def test_correpairs2rotation():
    """Get the 3D rotation from 3D points pairs."""
    # x = 90 degree
    src_points_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    tar_points_list = [[1, 0, 0], [0, 0, 1], [0, -1, 0], [0, 0, 0]]

    src_points = np.array(src_points_list)
    tar_points = np.array(tar_points_list)

    rotation_mat = pointcloud_utils.correpairs2rotation(src_points, tar_points)
    print(R.from_matrix(rotation_mat).as_euler("xyz", degrees=True))
    print(rotation_mat)


if __name__ == "__main__":
    test_correpairs2rotation()
