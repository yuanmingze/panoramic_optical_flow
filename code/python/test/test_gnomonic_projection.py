from re import X
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


def test_reverse_gnomonic_projection_hemisphere():
    tangent_point_theta = 0.0
    tangent_point_phi = 0.0

    tangent_x_list = np.array([1.0, 1.0])
    tangent_y_list = np.array([0.7, 0.7])
    hemisphere_list = np.array(np.array([True, False], np.bool))

    sph_list_theta, sph_list_phi = gp.reverse_gnomonic_projection(tangent_x_list, tangent_y_list, tangent_point_theta, tangent_point_phi, hemisphere_list)

    print("{}".format(sph_list_theta[0] - sph_list_theta[1]))

    for idx in range(sph_list_theta.shape[0]):
        print("Theta: {}, Phi: {} , SameHemiSph: {}".format(sph_list_theta[idx], sph_list_phi[idx], hemisphere_list[idx]))


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


def test_gnomonic_projection_hemisphere_index():
    """Test sph->GP with hemisphere index.
    """
    # tangent point
    theta_0 = np.radians(0.0)
    phi_0 = np.radians(0.0)

    # the point in sphere
    theta = np.radians(np.array([80.0, -100.0, 40.0, -140], np.float))
    phi = np.radians(np.array([0.0, 0.0, 20.0, -20.0], np.float))

    #
    x_g, y_g, overflow_label = gp.gnomonic_projection(theta, phi, theta_0, phi_0, True)
    print("Gnomonic Projection\nx: {} \ny: {} \n {}".format(x_g, y_g, overflow_label))


def test_gnomonic_unit():
    """Compute the tangent image's unit.
    The relationship between the tangent image and radius.
    Whether the projected point in tangent image is equal the gnomonic point.
    """
    theta_0 = np.radians(10.0)
    phi_0 = np.radians(5.0)

    theta = np.radians(np.array([10.0, 30.0, 45.0, 56.0], np.float))
    phi = np.radians(np.array([20.0, 45.0, 60.0, -10.0], np.float))

    # 1) compute the gnomonic projection
    x_g, y_g = gp.gnomonic_projection(theta, phi, theta_0, phi_0)
    print("1) Gnomonic Projection\nx: {} \ny: {}".format(x_g, y_g))

    # 0) compute the 3D projection
    x_t, y_t, z_t = gp.tangent3d_projection(theta, phi, theta_0, phi_0)
    print("2) Tangent 3D Projection\nx: {} \ny: {}".format(x_t, y_t))

    # error is large
    assert np.allclose(x_g, x_t) and np.allclose(y_g, y_t)


if __name__ == "__main__":
    # test_gnomonic_projection()
    # test_reverse_gnomonic_projection()
    test_gnomonic_unit()

    test_list =[0]

    if 0 in test_list:
        test_gnomonic_projection_hemisphere_index()

    if 1 in test_list:
        test_reverse_gnomonic_projection_hemisphere()
