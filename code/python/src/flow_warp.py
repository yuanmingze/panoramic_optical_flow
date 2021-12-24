import numpy as np
from scipy import ndimage
from scipy.spatial.transform import rotation
from scipy.stats import norm

from . import pointcloud_utils
from . import spherical_coordinates
from . import flow_postproc
from . import spherical_coordinates as sc
from . import flow_warp
from .logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def flow_warp_meshgrid(motion_flow_u, motion_flow_v):
    """
    warp the the original points (image's mesh grid) with the motion vector, meanwhile process the warp around.

    :param motion_flow_u: [height, width]
    :type motion_flow_u: numpy
    :param motion_flow_v: [height, width]
    :type motion_flow_v: numpy
    :return: the target points
    :rtype: numpy
    """
    if np.shape(motion_flow_u) != np.shape(motion_flow_v):
        log.error("motion flow u shape {} is not equal motion flow v shape {}".format(np.shape(motion_flow_u), np.shape(motion_flow_v)))

    # get the mesh grid
    height = np.shape(motion_flow_u)[0]
    width = np.shape(motion_flow_u)[1]
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    x_array, y_array = np.meshgrid(x_index, y_index)

    # get end point location
    end_points_u = x_array + motion_flow_u
    end_points_v = y_array + motion_flow_v

    # process the warp around
    u_index = end_points_u >= width - 0.5
    end_points_u[u_index] = end_points_u[u_index] - width
    u_index = end_points_u < -0.5
    end_points_u[u_index] = end_points_u[u_index] + width

    v_index = end_points_v >= height-0.5
    end_points_v[v_index] = end_points_v[v_index] - height
    v_index = end_points_v < -0.5
    end_points_v[v_index] = end_points_v[v_index] + height

    return np.stack((end_points_u, end_points_v))


def warp_backward(image_target, of_forward):
    """
    Backward warp with optical flow from the target image to generate the source image. 

    :param image_target: The terget image of optical flow, [height, width, channel].
    :type image_target: numpy
    :param of_forward:  optical flow from source to target, [height, width, 2].
    :type of_forward: numpy
    :return: Generated source image.
    :rtype: numpy
    """
    image_height = image_target.shape[0]
    image_width = image_target.shape[1]
    image_channels = None
    if len(image_target.shape) == 3:
        image_channels = image_target.shape[2]
    elif len(image_target.shape) == 2:
        image_channels = None
    else:
        log.error("The image shape is {}, do not support.".format(image_target.shape))
    dest_image = np.zeros_like(image_target, dtype=image_target.dtype)

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    # x_idx_new = np.remainder(x_idx + of_forward[:, :, 0], image_width)
    # y_idx_new = np.remainder(y_idx + of_forward[:, :, 1], image_height)
    x_idx_new = (x_idx + of_forward[:, :, 0])
    y_idx_new = (y_idx + of_forward[:, :, 1])

    if image_channels is not None:
        for channel_index in range(0, image_channels):
            dest_image[y_idx.astype(int), x_idx.astype(int), channel_index] = ndimage.map_coordinates(image_target[:, :, channel_index], [y_idx_new, x_idx_new], order=1, mode='wrap')
    else:
        dest_image[y_idx.astype(int), x_idx.astype(int)] = ndimage.map_coordinates(image_target[:, :], [y_idx_new, x_idx_new], order=1, mode='constant', cval=255)

    return dest_image


def warp_forward_padding(image_target, of_forward, padding_x=0, padding_y=0):
    '''
    warp the target image to the source image with the forward optical flow.
    The padding is used in the case the optical flow warp out of the image range.
    '''
    image_size = np.shape(image_target)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    image_src = 255 * np.ones((image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype=image_target.dtype)
    image_src[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 + padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 + padding_y * 2, image_height + padding_y * 2)

    of_forward_x = np.pad(of_forward[:, :, 0], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    x_idx = (x_idx_tar + of_forward_x).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    x_idx_tar = x_idx_tar.astype(np.int)
    x_idx_tar = x_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx_tar + of_forward_y).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    y_idx_tar = y_idx_tar.astype(np.int)
    y_idx_tar = y_idx_tar[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # check the range of x_idx & y_idx
    if not np.logical_and(x_idx_tar >= 0, x_idx_tar < image_width).all():
        print("image warp x_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return
    if not np.logical_and(y_idx_tar >= 0, y_idx_tar < image_height).all():
        print("image warp y_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return

    # 1) get new warpped image
    image_target_padded = np.pad(image_target, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_src[y_idx_tar, x_idx_tar] = image_target_padded[y_idx, x_idx]
    return image_src


def warp_forward(image_first, of_forward, wrap_around=False, ignore_transparent=False):
    """ forward warp image with optical flow. 

    :param image_first: input image, when it's 4 channels image, use the alpha channel to ignore the transparent area [height,width,:].
    :type image_first: numpy
    :param of_forward: forward optical flow. [height, width,  2]
    :type of_forward: numpy
    :param wrap_around: whether process the wrap around pixels, defaults to False
    :type wrap_around: bool, optional
    :param ignore_transparent: if yes do not warp the transparent are in the first image, defaults to False
    :type ignore_transparent: bool, optional
    :return: warped image
    :rtype: numpy
    """
    valid_pixels_index = None
    if image_first.shape[2] == 4:
        # RGBA images, ignore the transparent area
        valid_pixels_index = image_first[:, :, 3] == 255

    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    # x_idx, y_idx = np.mgrid[0:image_height, 0: image_width]
    x_idx_new = x_idx + of_forward[:, :, 0]
    y_idx_new = y_idx + of_forward[:, :, 1]

    # check index out of the image bounds
    if wrap_around:
        x_idx_new = np.where(x_idx_new > 0, x_idx_new, x_idx_new + image_width)
        x_idx_new = np.where(x_idx_new < image_width, x_idx_new, np.remainder(x_idx_new, image_width))
        y_idx_new = np.where(y_idx_new > 0, y_idx_new, y_idx_new + image_height)
        y_idx_new = np.where(y_idx_new < image_height, y_idx_new, np.remainder(y_idx_new, image_height))
    else:
        x_idx_new = np.where(x_idx_new > 0, x_idx_new, 0)
        x_idx_new = np.where(x_idx_new < image_width, x_idx_new, image_width - 1)
        y_idx_new = np.where(y_idx_new > 0, y_idx_new, 0)
        y_idx_new = np.where(y_idx_new < image_height, y_idx_new, image_height - 1)

    if valid_pixels_index is not None:
        x_idx_new = x_idx_new[valid_pixels_index]
        y_idx_new = y_idx_new[valid_pixels_index]
        x_idx = x_idx[valid_pixels_index]
        y_idx = y_idx[valid_pixels_index]

    x_idx_new = x_idx_new.astype(int)
    y_idx_new = y_idx_new.astype(int)
    # dest_image = np.zeros(np.shape(image_first), dtype=image_first.dtype)
    dest_image = np.full_like(image_first, [255, 255, 255], dtype=image_first.dtype)
    for channel_index in range(0, image_channels):
        dest_image[y_idx_new, x_idx_new, channel_index] = ndimage.map_coordinates(image_first[:, :, channel_index], [y_idx, x_idx], order=1, mode='constant', cval=255)

    return dest_image


def warp_forward_padding(image_first, of_forward, padding_x=0, padding_y=0):
    """
    forward warpping
    The padding to protect the pixel warped range out of image boundary
    """
    image_size = np.shape(image_first)
    image_height = image_size[0]
    image_width = image_size[1]
    image_channels = image_size[2]

    dest_image = 255 * np.ones((image_size[0] + padding_y * 2, image_size[1] + padding_x * 2, 3), dtype=image_first.dtype)
    dest_image[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)] = 0

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1 + padding_x * 2, image_width + padding_x * 2)
    y_idx_arr = np.linspace(0, image_height - 1 + padding_y * 2, image_height + padding_y * 2)

    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = np.pad(of_forward[:, :, 0], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))
    of_forward_y = np.pad(of_forward[:, :, 1], ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=(0))

    x_idx = (x_idx + of_forward_x).astype(np.int)
    x_idx = x_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # x_idx = np.clip(x_idx, 0, image_width + padding_x * 2 - 1)

    y_idx = (y_idx + of_forward_y).astype(np.int)
    y_idx = y_idx[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    # y_idx = np.clip(y_idx, 0, image_hight + padding_y * 2 - 1)

    # check the range of x_idx & y_idx
    if not np.logical_and(x_idx >= 0, x_idx < image_width).all():
        print("image warp x_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return
    if not np.logical_and(y_idx >= 0, y_idx < image_height).all():
        print("image warp y_idx out of range, max is {}, min is {}".format(x_idx.max(), x_idx.min()))
        return

    # 1) get new warpped image
    image_first_padded = np.pad(image_first, ((padding_y, padding_y), (padding_x, padding_x), (0, 0)), 'constant', constant_values=255)
    image_first_padded = image_first_padded[padding_y:(padding_y + image_height), padding_x:(padding_x + image_width)]
    dest_image[y_idx, x_idx] = image_first_padded
    return dest_image


def flow2rotation_3d(erp_flow, mask_method = "center"):
    """Compute the two image rotation from the ERP image's optical flow with SVD.
    The rotation is from the first image to second image.

    :param erp_flow: The ERP optical flow. [height, width,2]
    :type erp_flow: numpy
    :return: The rotation matrix.
    :rtype: numpy
    """
    # 0) source 3D points and target 3D points
    motion_flow_u = erp_flow[:, :, 0]
    motion_flow_v = erp_flow[:, :, 1]
    tar_points_2d = flow_warp_meshgrid(motion_flow_u, motion_flow_v)

    height = np.shape(motion_flow_u)[0]
    width = np.shape(motion_flow_u)[1]
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    x_array, y_array = np.meshgrid(x_index, y_index)
    src_points_2d = np.stack((x_array, y_array))

    # convert to 3D points
    src_points_2d_sph = sc.erp2sph(src_points_2d)
    tar_points_2d_sph = sc.erp2sph(tar_points_2d)

    if mask_method is "center":
        # just use the center rows optical flow
        row_idx_start = int(height * 0.25)
        row_idx_end = int(height * 0.75)
        src_points_2d_sph = src_points_2d_sph[:, row_idx_start:row_idx_end, :]
        tar_points_2d_sph = tar_points_2d_sph[:, row_idx_start:row_idx_end, :]

    src_points_3d = sc.sph2car(src_points_2d_sph[0], src_points_2d_sph[1])
    tar_points_3d = sc.sph2car(tar_points_2d_sph[0], tar_points_2d_sph[1])

    # 1) SVD get the rotation matrix
    src_points_3d = np.swapaxes(src_points_3d.reshape((3, -1)), 0, 1)
    tar_points_3d = np.swapaxes(tar_points_3d.reshape((3, -1)), 0, 1)
    rotation_mat = pointcloud_utils.correpairs2rotation(src_points_3d, tar_points_3d)

    return rotation_mat


def flow2rotation_2d(erp_flow_, use_weight=True):
    """Compute the  two image rotation from the ERP image's optical flow.
    The rotation is from the first image to second image.

    :param erp_flow: the erp image's flow 
    :type erp_flow: numpy 
    :param use_weight: use the centre rows and columns to compute the rotation, default is True.
    :type: bool
    :return: the offset of ERP image, [theta shift, phi shift], radian
    :rtype: float
    """
    erp_image_height = erp_flow_.shape[0]
    erp_image_width = erp_flow_.shape[1]

    # convert the pixel offset to rotation radian
    erp_flow = flow_postproc.erp_of_wraparound(erp_flow_)
    theta_delta_array = 2.0 * np.pi * (erp_flow[:, :, 0] / erp_image_width)
    theta_delta = np.mean(theta_delta_array)

    # just the center column of the optical flow.
    delta = theta_delta / (2.0 * np.pi)
    flow_col_start = int(erp_image_width * (0.5 - delta))
    flow_col_end = int(erp_image_width * (0.5 + delta))
    if delta < 0:
        temp = flow_col_start
        flow_col_start = flow_col_end
        flow_col_end = temp
    flow_col_center = np.full((erp_image_height, erp_image_width), False, dtype=np.bool)
    flow_col_center[:, flow_col_start:flow_col_end] = True
    flow_sign = np.sign(np.sum(np.sign(erp_flow[flow_col_center, 1])))
    # phi_delta_array = np.pi * (erp_flow[flow_col_start:flow_col_end, :, 1] / erp_image_height)
    if flow_sign < 0:
        positive_index = np.logical_and(erp_flow[:, :, 1] < 0, flow_col_center)
    else:
        positive_index = np.logical_and(erp_flow[:, :, 1] > 0, flow_col_center)
    phi_delta_array = -np.pi * (erp_flow[positive_index, 1] / erp_image_height)

    if use_weight:
        # TODO Check the weight performance
        # weight of the u, width
        stdev = erp_image_height * 0.5 * 0.25
        weight_u_array_index = np.arange(erp_image_height)
        weight_u_array = norm.pdf(weight_u_array_index, erp_image_height / 2.0, stdev)
        theta_delta_array = np.average(theta_delta_array, axis=0, weights=weight_u_array)

        # weight of the v, height
        stdev = erp_image_width * 0.5 * 0.25
        weight_v_array_index = np.arange(erp_image_width)
        weight_v_array = norm.pdf(weight_v_array_index, erp_image_width / 2.0, stdev)
        phi_delta_array = np.average(phi_delta_array, axis=1,  weights=weight_v_array)

    phi_delta = np.mean(phi_delta_array)

    return theta_delta, phi_delta


def global_rotation_warping(erp_image, erp_flow, forward_warp=True, rotation_type="3D"):
    """ Global rotation warping.

    Rotate the ERP image base on the flow. 
    If `forward_warp` is True, the `erp_image` is the source image, `erp_flow` is form source to target.
    If `forward_warp` is False, the `erp_image` is the target image, `erp_flow` is from source to target.

    :param erp_image: the image of optical flow, 
    :type erp_image: numpy 
    :param erp_flow: the erp image's flow, from source image to target image.
    :type erp_flow: numpy 
    :param forward_warp: If yes, the erp_image is use the erp_flow forward warp erp_image.
    :type forward_warp: bool 
    :param 
    :return: The rotated ERP image,  the rotation matrix from original to target (returned erp image).
    :rtype: numpy
    """
    # 0) get the rotation matrix from optical flow
    if rotation_type == "2D":
        # compute the average of optical flow & get the delta theta and phi
        theta_delta, phi_delta = flow2rotation_2d(erp_flow, False)

        if not forward_warp:
            theta_delta = -theta_delta
            phi_delta = -phi_delta
        rotation_mat = spherical_coordinates.rot_sph2mat(theta_delta, phi_delta, False)
    elif rotation_type == "3D":
        rotation_mat = flow2rotation_3d(erp_flow)
        if not forward_warp:
            rotation_mat = rotation_mat.T
    else:
        log.error("Do not suport rotation type {}".format(rotation_type))

    # 1) rotate the ERP image with the rotation matrix
    erp_image_rot = sc.rotate_erp_array(erp_image, rotation_mat = rotation_mat)
    if erp_image.dtype == np.uint8:
        erp_image_rot = erp_image_rot.astype(np.uint8)

    return erp_image_rot, rotation_mat
