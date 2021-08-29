import configuration as config

from utility import pointcloud_utils
from scipy.spatial.transform import Rotation as R

import numpy as np

def test_correpairs2rotation():
    """Get the 3D rotation from 3D points pairs."""
    src_points_list = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    src_points = np.array(src_points_list)

    tar_points_list = [[1,0,0],[0,0,1],[0,-1,0],[0,0,0]]
    tar_points = np.array(tar_points_list)

    rotation_mat = pointcloud_utils.correpairs2rotation(src_points, tar_points)
    print(R.from_matrix(rotation_mat).as_euler("xyz", degrees=True))
    print(rotation_mat)


if __name__ == "__main__":
    test_correpairs2rotation()
