import configuration as config

from utility import gnomonic_projection as gp
import numpy as np
import matplotlib.pyplot as plt


def test_reverse_gnomonic_projection():
    """Test reverse gnomonic projection
    """
    # # the ERP image seam
    # tangent_point_theta = np.pi
    # tangent_point_phi = 0

    # # the north pole
    # tangent_point_theta = 0
    # tangent_point_phi = np.pi / 2.0

    # the north pole
    tangent_point_theta = 0.0
    tangent_point_phi = 0.0

    # tangent_list = np.array.zeros((3,2), dtype = np.float)
    tangent_x_list = np.linspace(-0.8, 0.8, num=40)
    tangent_y_list = np.linspace(-1.0, 1.0, num=40)
    tangent_x, tangent_y = np.meshgrid(tangent_x_list, tangent_y_list)

    sph_list_x, sph_list_y = gp.reverse_gnomonic_projection(tangent_x, tangent_y, tangent_point_theta, tangent_point_phi)

    tangent_point_list = np.stack((tangent_x.flatten(), tangent_y.flatten()), axis=1)
    print("tangetn_point_list \n {}".format(tangent_point_list))
    print("spherical_point_list \n {}".format(np.stack((sph_list_x.flatten(), sph_list_y.flatten()), axis=1)))

    color = range(len(tangent_x.flatten()))
    cm = plt.cm.get_cmap('RdYlBu')
    plt.subplot(221)
    plt.ylabel('tangent image points')
    plt.scatter(tangent_x.flatten(), tangent_y.flatten(), c=color, marker='o', cmap=cm)

    plt.subplot(222)
    plt.ylabel('spherical coordinate image points')
    plt.scatter(sph_list_x, sph_list_y,  c=color, marker='o', cmap=cm)

    plt.show()


def test_gnomonic_projection():
    """Test reverse gnomonic projection
    """
    # # the ERP image seam
    # tangent_point_theta = np.pi
    # tangent_point_phi = 0

    # # the north pole
    # tangent_point_theta = 0
    # tangent_point_phi = np.pi / 2.0

    # the north pole
    # tangent_point_theta = 0.0
    # tangent_point_phi = 0.0
    tangent_point_theta = 0.3
    tangent_point_phi = -0.4

    # tangent_list = np.array.zeros((3,2), dtype = np.float)
    tangent_x_list = np.linspace(-0.5, 0.5, num=40)
    tangent_y_list = np.linspace(-0.4, 0.4, num=40)
    tangent_x, tangent_y = np.meshgrid(tangent_x_list, tangent_y_list)

    sph_list_x, sph_list_y = gp.gnomonic_projection(tangent_x, tangent_y, tangent_point_theta, tangent_point_phi)

    tangent_point_list = np.stack((tangent_x.flatten(), tangent_y.flatten()), axis=1)
    print("tangetn_point_list \n {}".format(tangent_point_list))
    print("spherical_point_list \n {}".format(np.stack((sph_list_x.flatten(), sph_list_y.flatten()), axis=1)))

    color = range(len(tangent_x.flatten()))
    cm = plt.cm.get_cmap('RdYlBu')
    plt.subplot(221)
    plt.ylabel('spherical coordinate image points')
    plt.scatter(tangent_x.flatten(), tangent_y.flatten(), c=color, marker='o', cmap=cm)

    plt.subplot(222)
    plt.ylabel('tangent image points')
    plt.scatter(sph_list_x, sph_list_y,  c=color, marker='o', cmap=cm)

    plt.show()


if __name__ == "__main__":
    test_gnomonic_projection()
    test_reverse_gnomonic_projection()
