import numpy as np
from scipy.spatial.transform import Rotation as R

from . import flow_warp

from .logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def great_circle_distance(points_1, points_2, radius=1):
    """
    Get the distance between two points in the sphere, the grate-circle distance.
    Reference: https://en.wikipedia.org/wiki/Great-circle_distance.

    :param points_1: the numpy array [theta_1, phi_1]
    :type points_1: numpy
    :param points_2: the numpy array [theta_2, phi_2]
    :type points_2: numpy
    :param radius: the radius, the default is 1
    :return distance: the great-circle distance of two point.
    :rtype: numpy
    """
    return great_circle_distance_uv(points_1[0], points_1[1], points_2[0], points_2[1], radius)


def great_circle_distance_uv(points_1_theta, points_1_phi, points_2_theta, points_2_phi, radius=1):
    """
    @see great_circle_distance (haversine distances )
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html

    :param points_1_theta: theta in radians, size is [N]
    :type points_1_theta : numpy
    :param points_1_phi: phi in radians, size is [N]
    :type points_1_phi : numpy
    :param points_2_theta: radians
    :type points_2_theta: float
    :param points_2_phi: radians
    :type points_2_phi: float
    :return: The geodestic distance from point ot tangent point.
    :rtype: numpy
    """
    delta_theta = points_2_theta - points_1_theta
    delta_phi = points_2_phi - points_1_phi
    a = np.sin(delta_phi * 0.5) ** 2 + np.cos(points_1_phi) * np.cos(points_2_phi) * np.sin(delta_theta * 0.5) ** 2
    central_angle_delta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    if np.isnan(central_angle_delta).any():
        log.warn("the circle angle have NAN")

    return np.abs(radius * central_angle_delta)


def get_angle_sph_ofcolor(points_center, points_u, points_v):
    """
    A triangle's angle between center_point, u and v on the surface of a sphere.
    The angle range is [0, 2*pi], and angle is from cu to cv on clock-wise direction.

    :param points_center: The pixel point location, The (theta, phi) radian of point A on sphere, [point_number, 2].
    :type points_center: numpy
    :param points_u: The U (horizontal), theta part of optical flow, [point_number, 2].
    :type points_u: numpy
    :param points_v: The V (vertical), phi part of the optical flow, [point_number, 2].
    :type points_v: numpy
    :return: the angle between cu and uv, radian
    :rtype: numpy
    """
    # module the theta and phi

    # 1) get the quadrants
    #TODO process the u or v equal 0
    u_sign_positive_0 = points_u[:, 0] > points_center[:, 0]
    u_sign_positive_1 = np.logical_and(points_center[:, 0] > 0, points_u[:, 0] < 0)
    u_sign_positive_1 = np.logical_and(u_sign_positive_1, ((np.pi - points_center[:, 0]) + (points_u[:, 0] - (-np.pi))) < np.pi)
    u_sign_positive = np.logical_or(u_sign_positive_0, u_sign_positive_1)

    # u_axis_positive = np.logical_and(points_v[:, 1] == points_center[:, 1], u_sign_positive_1)
    # u_axis_negative = np.logical_and(points_v[:, 1] == points_center[:, 1], ~u_sign_positive_1)

    v_sign_positive_0 = points_v[:, 1] > points_center[:, 1]
    v_sign_positive_1 = np.logical_and(points_center[:, 1] > 0, points_v[:, 1] < 0)
    v_sign_positive_1 = np.logical_and(v_sign_positive_1,  ((0.5*np.pi - points_center[:, 0]) + (0.5 * np.pi + points_v[:, 0])) < 0.5 * np.pi)
    v_sign_positive = np.logical_or(v_sign_positive_0, v_sign_positive_1)

    # v_axis_positive = np.logical_and(points_u[:, 0] == points_center[:, 0], v_sign_positive)
    # v_axis_negative = np.logical_and(points_u[:, 0] == points_center[:, 0], ~v_sign_positive)

    quadrants_index = np.zeros_like(u_sign_positive, dtype=np.int64)
    quadrants_index[np.logical_and(v_sign_positive, u_sign_positive)] = 1
    quadrants_index[np.logical_and(v_sign_positive, ~u_sign_positive)] = 2
    quadrants_index[np.logical_and(~v_sign_positive, ~u_sign_positive)] = 3
    quadrants_index[np.logical_and(~v_sign_positive, u_sign_positive)] = 4

    # 2) get the angle
    length_cu = great_circle_distance_uv(points_center[:, 0], points_center[:, 1], points_u[:, 0], points_u[:, 1], radius=1)
    length_cv = great_circle_distance_uv(points_center[:, 0], points_center[:, 1], points_v[:, 0], points_v[:, 1], radius=1)
    length_uv = great_circle_distance_uv(points_u[:, 0], points_u[:, 1], points_v[:, 0], points_v[:, 1], radius=1)

    # with tangent two avoid the error of Float
    s = 0.5 * (length_uv + length_cv + length_cu)
    numerator = np.sin(s-length_cu) * np.sin(s-length_uv)
    denominator = np.sin(s) * np.sin(s-length_cv)
    denominator_nan_idx = None
    if (denominator == 0).any():
        # log.warn("The angle_A contain NAN")
        denominator_nan_idx = denominator == 0
        denominator[denominator_nan_idx] = pow(0.1, 10)
    temp_data = numerator / denominator
    angle_cv = 2 * np.arctan(np.sqrt(np.abs(temp_data)))
    if denominator_nan_idx is not None:
        angle_cv[denominator_nan_idx] = np.pi * 0.5
    # angle_A = 2*np.pi - angle_A # second solution

    # # check constraints
    # indices_1 = (np.abs(np.pi - length_cu) - np.abs(np.pi-length_cv)) > np.abs(np.pi-length_uv)
    # indices_2 = (np.abs(np.pi-length_uv) > (np.abs(np.pi-length_cu) + np.abs(np.pi-length_cv)))
    # indices = np.logical_or(indices_1, indices_2)
    # if indices.any():
    #     log.warn("side length check constraints wrong.")
    #     angle_cv[indices] = 2*np.pi - angle_cv[indices]

    # # 3) correct the quadrants, increasing clockwise and start from +u.
    # angle_cv[np.logical_and(points_u[:, 0] == points_center[:, 0], points_v[:, 1] == points_center[:, 1])] = 0

    # angle_cv[u_axis_positive] = 0
    # angle_cv[v_axis_negative] = np.pi * 0.5
    # angle_cv[u_axis_negative] = np.pi
    # angle_cv[v_axis_positive] = np.pi * 1.5

    angle_cv[quadrants_index == 3] = -angle_cv[quadrants_index == 3] + np.pi
    angle_cv[quadrants_index == 2] = angle_cv[quadrants_index == 2] + np.pi
    angle_cv[quadrants_index == 1] = -angle_cv[quadrants_index == 1] + 2.0 * np.pi

    # if the center at pole the angle is 
    poles_index = np.logical_or(points_center[:,1] == np.pi * 0.5, points_center[:,1] == -np.pi * 0.5)
    if poles_index.any():
        angle_cv[poles_index] = points_u[poles_index,0]

    return angle_cv


def get_angle_sph(points_A, points_B, points_C):
    """
    A triangle's angle between AB and AC on the surface of a sphere.
    
    https://mathworld.wolfram.com/SphericalTrigonometry.html
    https://en.wikipedia.org/wiki/Spherical_trigonometry

    Half-angle and half-side formule: 
    https://en.wikipedia.org/wiki/Solution_of_triangles#Solving_spherical_triangles

  
    :param points_A: The (theta, phi) radian of point A on sphere, [point_number, 2]
    :type points_A: numpy
    :param points_B: The (theta, phi) radian of point B on sphere, [point_number, 2]
    :type points_B: list or tuple
    :param points_C: The (theta, phi) radian of point C on sphere, [point_number, 2]
    :type points_C: numpy
    :return: the angle between AB and AC, alway return the smaller angle which is in range [0, +pi]
    :rtype: numpy
    """
    points_A_theta = points_A[:,0]
    points_A_phi = points_A[:,1]
    points_B_theta = points_B[:,0]
    points_B_phi = points_B[:,1]
    points_C_theta = points_C[:,0]
    points_C_phi = points_C[:,1]

    length_AB = great_circle_distance_uv(points_A_theta, points_A_phi, points_B_theta, points_B_phi, radius=1)
    length_AC = great_circle_distance_uv(points_A_theta, points_A_phi, points_C_theta, points_C_phi, radius=1)
    length_BC = great_circle_distance_uv(points_B_theta, points_B_phi, points_C_theta, points_C_phi, radius=1)

    # # with the arccos
    # data = (np.cos(length_BC) - np.cos(length_AC) * np.cos(length_AB)) / (np.sin(length_AC) * np.sin(length_AB))
    # # remove the float error
    # if (np.abs(data) > (1.0 + np.finfo(float).eps * 100)).any():
    #     raise RuntimeError("the angle_A is not in range [-1,1]")
    # else:
    #     # data[data > 1.0] = 1.0
    #     if data > 1.0:
    #         data = 1.0
    #     # data[data < -1.0] = -1.0
    #     if data < -1.0:
    #         data = -1.0
    # angle_A = np.arccos(data)

    # with tangent two avoid the error of Float
    s = 0.5 * (length_BC + length_AC + length_AB)
    numerator = np.sin(s-length_AC) * np.sin(s-length_AB)
    denominator = np.sin(s) * np.sin(s-length_BC)
    if (denominator == 0).any():
        log.warn("The angle_A contain NAN")
        denominator[denominator == 0] = pow(0.1, 10)
    temp_data = numerator / denominator
    angle_A = 2 * np.arctan(np.sqrt(np.abs(temp_data)))
    # angle_A = 2*np.pi - angle_A # second solution

    # # check constraints
    # indices_1 = (np.abs(np.pi - length_AB) - np.abs(np.pi-length_AC)) > np.abs(np.pi-length_BC)
    # indices_2 = (np.abs(np.pi-length_BC) > (np.abs(np.pi-length_AB) + np.abs(np.pi-length_AC)))
    # indices = np.logical_or(indices_1, indices_2)
    # if indices.any():
    #     log.warn("side length check constraints wrong.")
    #     print(indices)
    #     angle_A[indices] = 2*np.pi - angle_A[indices]
    return angle_A


def erp_pixel_modulo_0(erp_points_list, image_height):
    """[summary]

    :param erp_points_list: The erp pixel list, [2, points_number]
    :type erp_points_list: numpy
    :param image_height: erp image height
    :type image_height: numpy
    """
    x = erp_points_list[0,:]
    y = erp_points_list[1,:]
    x, y = erp_pixel_modulo(x, y , image_height)
    return np.stack((x,y), axis=0)


def erp_pixel_modulo(x_arrray, y_array, image_height):
    """ Make x,y and ERP pixels coordinate system range.
    """
    image_width = 2 * image_height
    x_arrray_new = np.remainder(x_arrray + 0.5, image_width) - 0.5
    y_array_new = np.remainder(y_array + 0.5, image_height) - 0.5
    return x_arrray_new, y_array_new


def erp_sph_modulo(theta, phi):
    """Modulo of the spherical coordinate for the erp coordinate.
    """
    points_theta = np.remainder(theta + np.pi, 2 * np.pi) - np.pi
    points_phi = -(np.remainder(-phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi)
    return points_theta, points_phi


def erp2sph(erp_points, erp_image_height=None, sph_modulo=False):
    """
    convert the point from erp image pixel location to spherical coordinate.
    The image center is spherical coordinate origin.

    :param erp_points: the point location in ERP image x∊[0, width-1], y∊[0, height-1] , size is [2, :]
    :type erp_points: numpy
    :param erp_image_height: ERP image's height, defaults to None
    :type erp_image_height: int, optional
    :param sph_modulo: if true, process the input points wrap around, .
    :type sph_modulo: bool
    :return: the spherical coordinate points, theta is in the range [-pi, +pi), and phi is in the range [-pi/2, pi/2)
    :rtype: numpy
    """
    # 0) the ERP image size
    if erp_image_height == None:
        height = np.shape(erp_points)[1]
        width = np.shape(erp_points)[2]

        if (height * 2) != width:
            log.error("the ERP image width {} is not two time of height {}".format(width, height))
    else:
        height = erp_image_height
        width = height * 2

    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]

    # 1) point location to theta and phi
    points_theta = erp_points_x * (2 * np.pi / width) + np.pi / width - np.pi
    points_phi = -(erp_points_y * (np.pi / height) + np.pi / height * 0.5) + 0.5 * np.pi

    if sph_modulo:
        points_theta, points_phi = erp_sph_modulo(points_theta, points_phi)

    points_theta = np.where(points_theta == np.pi,  -np.pi, points_theta)
    points_phi = np.where(points_phi == -0.5 * np.pi, 0.5 * np.pi, points_phi)

    return np.stack((points_theta, points_phi))


def sph2erp_0(sph_points, erp_image_height=None, sph_modulo=False):
    theta = sph_points[0, :]
    phi = sph_points[1, :]
    erp_x, erp_y = sph2erp(theta, phi, erp_image_height, sph_modulo)
    return np.stack((erp_x, erp_y), axis=0)


def sph2erp(theta, phi, erp_image_height, sph_modulo=False):
    """ 
    Transform the spherical coordinate location to ERP image pixel location.

    :param theta: longitude is radian
    :type theta: numpy
    :param phi: latitude is radian
    :type phi: numpy
    :param image_height: the height of the ERP image. the image width is 2 times of image height
    :type image_height: [type]
    :param sph_modulo: if yes process the wrap around case, if no do not.
    :type sph_modulo: bool, optional
    :return: the pixel location in the ERP image.
    :rtype: numpy
    """
    if sph_modulo:
        theta, phi = erp_sph_modulo(theta, phi)

    erp_image_width = 2 * erp_image_height
    erp_x = (theta + np.pi) / (2.0 * np.pi / erp_image_width) - 0.5
    erp_y = (-phi + 0.5 * np.pi) / (np.pi / erp_image_height) - 0.5
    return erp_x, erp_y



def car2sph(points_car, min_radius=1e-10):
    """
    Transform the 3D point from cartesian to unit spherical coordinate.

    :param points_car: The 3D point array, is [point_number, 3], first column is x, second is y, third is z
    :type points_car: numpy
    :return: the points spherical coordinate, (theta, phi)
    :rtype: numpy
    """
    radius = np.linalg.norm(points_car, axis=1)

    valid_list = radius > min_radius  # set the 0 radius to origin.

    theta = np.zeros((points_car.shape[0]), np.float)
    theta[valid_list] = np.arctan2(points_car[:, 0][valid_list], points_car[:, 2][valid_list])

    phi = np.zeros((points_car.shape[0]), np.float)
    phi[valid_list] = -np.arcsin(np.divide(points_car[:, 1][valid_list], radius[valid_list]))

    return np.stack((theta, phi), axis=1)


def sph2car(theta, phi, radius=1.0):
    """
    Transform the spherical coordinate to cartesian 3D point.

    :param theta: longitude
    :type theta: numpy
    :param phi: latitude
    :type phi: numpy
    :param radius: the radius of projection sphere
    :type radius: float
    :return: +x right, +y down, +z is froward
    :rtype: numpy
    """
    # points_cartesian_3d = np.array.zeros((theta.shape[0],3),np.float)
    x = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.cos(theta)
    y = -radius * np.sin(phi)

    return np.stack((x, y, z), axis=0)


def rotation_erp_horizontal_fast(input_image, horizon_degree):
    """ Rotate the ERP image along the X-axis (horizontal).

    :param input_image: image shape is [height, width, channels_nubmer]
    :type input_image: numpy
    :param horizon_degree: rotation degree along the +x direction.
    :type horizon_degree: float
    """
    image_width = input_image.shape[1]
    phi_roll_numb = horizon_degree / 360.0 * image_width
    # TODO it need interpolation
    input_image_rotated = np.roll(input_image, int(phi_roll_numb), axis=1)
    return input_image_rotated


def rotate_erp_array(erp_image, rotation_mat=None):
    """ Rotate the ERP image with the theta and phi.

    :param erp_image: The ERP image, [height, width, 3]
    :type erp_image: numpy
    :param rotation_theta: The source to target rotation's theta.
    :type rotation_theta: float
    :param rotation_phi: The source to target rotation's phi.
    :type rotation_phi: float
    """
    # flow from tar to src
    opticalflow = rotation2erp_motion_vector(erp_image.shape[0:2], rotation_matrix=rotation_mat.T)
    return flow_warp.warp_backward(erp_image, opticalflow)    # the image backword warp
    

def rotate_erp_array_skylib(data_array, rotation_matrix):
    """
    Rotate the image along the theta and phi.

    Note Skylib coordinate system x is left, Y is up, z is forward
    It is different with our method.

    :param data_array: the data array (image, depth map, etc.), size is [height, height*2, :]
    :type data_array: numpy
    :param rotate_theta: rotate along the longitude, radian
    :type rotate_theta: float
    :param rotate_phi: rotate along the latitude, radian
    :type rotate_phi: float 
    :return: the rotated data array
    :rtype: numpy
    """
    # The envmap's 3d coordinate system is +x right, +y up and -z front.
    # rotation_matrix = R.from_euler("xyz", [np.degrees(-rotate_phi), np.degrees(rotate_theta), 0], degrees=True).as_matrix()
    # rotation_matrix = sc.rot_sph2mat(rotate_theta, rotate_phi, False)
    # TODO get the coordinate system tranformation.
    cs_transfrom = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    log.warn("skylib coordinate system is differnet with use not fix yet.")
    rotation_matrix = cs_transfrom.dot(rotation_matrix)
    # rotate the ERP image
    from envmap import EnvironmentMap
    envmap = EnvironmentMap(data_array, format_='latlong')
    data_array_rot = envmap.rotate("DCM", rotation_matrix).data
    return data_array_rot


def rotate_sph_coord(sph_theta, sph_phi, rotate_theta=None, rotate_phi=None, rotation_matrix_=None):
    """ Rotate the sph spherical coordinate with the rotation (theta, phi).

    :param sph_theta: The spherical coordinate's theta array.
    :type sph_theta: numpy
    :param sph_phi: The spherical coordinate's phi array.
    :type sph_phi: numpy
    :param rotate_theta: The rotation along the theta, radians
    :type rotate_theta: float
    :param rotate_phi: The rotation along the phi, radians
    :type rotate_phi: float
    :return: Target points theta and phi array.
    :rtype: tuple
    """
    xyz = sph2car(sph_theta, sph_phi, radius=1.0)
    if rotation_matrix_ is None and rotate_theta is not None and rotate_phi is not None:
        rotation_matrix = R.from_euler("xyz", [rotate_phi, rotate_theta, 0], degrees=False).as_matrix()
    elif rotation_matrix_ is not None and rotate_theta is None and rotate_phi is None:
        rotation_matrix = rotation_matrix_
    else:
        log.error("Do not specify the rotation both matrix and (theta, phi).")
    xyz_rot = np.dot(rotation_matrix, xyz.reshape((3, -1)))
    array_xy_rot = car2sph(xyz_rot.T).T
    return array_xy_rot[0, :], array_xy_rot[1, :]


def rotation2erp_motion_vector(array_size, rotation_matrix=None, wraparound=False):
    """
    Convert the spherical coordinate rotation to ERP coordinate motion flow.
    With rotate the image's mesh grid.

    :param data_array: the array size, [array_hight, array_width]
    :type data_array: list
    :param rotate_theta: rotate along the longitude, radian
    :type rotate_theta: float
    :param rotate_phi: rotate along the latitude, radian
    :type rotate_phi: float
    """
    # 1) generage spherical coordinate for each pixel
    erp_x = np.linspace(0, array_size[1], array_size[1], endpoint=False)
    erp_y = np.linspace(0, array_size[0], array_size[0], endpoint=False)
    erp_vx, erp_vy = np.meshgrid(erp_x, erp_y)

    # 1) spherical system to Cartesian system and rotate the points
    sph_xy = erp2sph(np.stack((erp_vx, erp_vy)), erp_image_height=array_size[0], sph_modulo=False)
    # erp_x_rot, erp_y_rot = sph2erp(sph_xy[0, :], sph_xy[1, :], array_size[0], erp_modulo=False)
    xyz = sph2car(sph_xy[0], sph_xy[1], radius=1.0)
    xyz_rot = np.dot(rotation_matrix, xyz.reshape((3, -1)))
    array_xy_rot = car2sph(xyz_rot.T).T
    erp_x_rot, erp_y_rot = sph2erp(array_xy_rot[0, :], array_xy_rot[1, :], array_size[0], sph_modulo=False)

    # get motion vector
    motion_vector_x = erp_x_rot.reshape((array_size[0], array_size[1])) - erp_vx
    motion_vector_y = erp_y_rot.reshape((array_size[0], array_size[1])) - erp_vy

    # to wraparound optical flow
    if wraparound:
        cross_minus2pi, cross_plus2pi = sph_wraparound(sph_xy[0], array_xy_rot[0, :].reshape((array_size[0], array_size[1])))
        motion_vector_x[cross_minus2pi] = motion_vector_x[cross_minus2pi] - array_size[1]
        motion_vector_x[cross_plus2pi] = motion_vector_x[cross_plus2pi] + array_size[1]

    return np.stack((motion_vector_x, motion_vector_y), -1)


def rot_sph2mat(theta, phi, degrees_=True):
    """Convert the spherical rotation to rotation matrix.
    """
    return R.from_euler("xyz", [phi, theta, 0], degrees=degrees_).as_matrix()


def rot_mat2sph(rot_mat, degrees_ = True):
    """Convert the 3D rotation to spherical coodinate theta and phi."""
    euler_angle = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees_)
    # log.debug("rot_mat2sph: {}".format(euler_angle))
    return tuple(euler_angle[:2])


def sph_coord_modulo(theta, phi):
    """Modulo the spherical coordinate.
    """
    theta_new = np.remainder(theta + np.pi, np.pi * 2.0) - np.pi
    phi_new = np.remainder(phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    return theta_new, phi_new


def sph_wraparound(src_theta, tar_theta):
    """ Get the line index cross the ERP image boundary.

    :param src_theta: The source point theta.
    :type src_theta: numpy
    :param tar_theta: The target point theta.
    :type tar_theta: numpy
    :return: Pixel index of line wrap around.
    :rtype: numpy
    """
    # face_src_x_sph_avail = face_src_xy_sph[0, :, :][available_list]
    # face_src_y_sph_avail = face_src_xy_sph[1, :, :][available_list]
    # face_src_x_sph_avail, face_src_y_sph_avail = sc.sph_coord_modulo(face_src_x_sph_avail, face_src_y_sph_avail)

    long_line = np.abs(tar_theta - src_theta) > np.pi
    cross_minus2pi = np.logical_and(src_theta < 0, tar_theta > 0)
    cross_minus2pi = np.logical_and(long_line, cross_minus2pi)
    cross_plus2pi = np.logical_and(src_theta > 0, tar_theta < 0)
    cross_plus2pi = np.logical_and(long_line, cross_plus2pi)
    # face_tar_x_erp[cross_x_axis_minus2pi] = face_tar_x_erp[cross_x_axis_minus2pi] - erp_flow_width
    # face_tar_x_erp[cross_x_axis_plus2pi] = face_tar_x_erp[cross_x_axis_plus2pi] + erp_flow_width

    return cross_minus2pi, cross_plus2pi
