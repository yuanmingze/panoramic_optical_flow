

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from . import spherical_coordinates


"""
functions used to evaluate the quality of optical flow;
"""

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
# LARGEFLOW = 1e8


def error_visual(error_data, max=None, min=None, verbose=False):
    """
    visualize the error data, and return colored error image.

    :param spherical: whether compute the EPE in spherical coordinate, 
    :param verbose: show the error images
    """
    if max is None:
        max = np.max(error_data)
    if min is None:
        min = np.min(error_data)
    if verbose:
        print("error_visual(): max error {}, min error {}".format(max, min))
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cmap = plt.get_cmap('jet')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return (m.to_rgba(error_data)[:, :, :3] * 255).astype(np.uint8)


def available_pixel(flow, unknown_value=UNKNOWN_FLOW_THRESH):
    """
    The criterion of the available optical flow pixel.
    1) not unknown value (very large or NaN)

    :return: available pixel index, a numpy boolon array, True is valid and False is invalid
    """
    min_value = -1

    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]

    # unknown pixel optical flow
    index_unknown = (abs(flow_u) > unknown_value) | (abs(flow_v) > unknown_value)
    index_know = np.logical_not(index_unknown)

    # zero optical flow
    index_unzero = (np.absolute(flow_u) > min_value) | (np.absolute(flow_v) > min_value)

    # valid pixels index
    index_valid = np.logical_and(index_know, index_unzero)
    return index_valid


def EPE(of_ground_truth, of_evaluation, spherical=False):
    """
    endpoint error (EE)
    reference : https://github.com/prgumd/GapFlyt/blob/master/Code/flownet2-tf-umd/src/flowlib.py

    :param spherical: whether compute the EPE in spherical coordinate, 
    return: average of EPE
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # remove the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    # index_su = su[ind2]
    # index_sv = sv[ind2]
    # # an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    # # un = index_su * an
    # # vn = index_sv * an

    # index_stu = stu[ind2]
    # index_stv = stv[ind2]
    # # tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    # # tun = index_stu * tn
    # # tvn = index_stv * tn

    # # angle = un * tun + vn * tvn + (an * tn)
    # # index = [angle == 1.0]
    # # angle[index] = 0.999
    # # ang = np.arccos(angle)
    # # mang = np.mean(ang)
    # # mang = mang * 180 / np.pi

    if spherical:
        # compute end point
        height = np.shape(stu)[0]
        width = np.shape(stu)[1]
        x_index = np.linspace(0, width - 1, width)
        y_index = np.linspace(0, height - 1, height)
        x_array, y_array = np.meshgrid(x_index, y_index)

        end_points_gt_u = x_array + stu
        end_points_gt_v = y_array + stv
        end_points_eva_u = x_array + su
        end_points_eva_v = y_array + sv

        # end point location to theta and phi
        end_points_gt_u = (end_points_eva_u - width / 2.0) / (width / 2.0) * np.pi
        end_points_gt_v = -(end_points_eva_v - height / 2.0) / (height / 2.0) * (np.pi / 2.0)

        end_points_eva_u = (end_points_eva_u - width / 2.0) / (width / 2.0) * np.pi
        end_points_eva_v = -(end_points_eva_v - height / 2.0) / (height / 2.0) * (np.pi / 2.0)

        of_gt_endpoints = spherical_coordinates.flow_warp_meshgrid(stu, stv)
        of_gt_endpoints_uv = spherical_coordinates.erp2spherical(of_gt_endpoints)

        of_eva_endpoints = spherical_coordinates.flow_warp_meshgrid(su, sv)
        of_eva_endpoints_uv = spherical_coordinates.erp2spherical(of_eva_endpoints)

        # get great circle distance
        epe = spherical_coordinates.great_circle_distance(of_gt_endpoints_uv, of_eva_endpoints_uv)
    else:
        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)

    epe = epe[ind2]
    mepe = np.mean(epe)

    return mepe


def EPE_mat(of_ground_truth, of_evaluation, spherical=False):
    """
    endpoint error (EE)
    reference : https://github.com/prgumd/GapFlyt/blob/master/Code/flownet2-tf-umd/src/flowlib.py

    return: average of EPE data 
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # remove the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    # ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    # index_su = su[ind2]
    # index_sv = sv[ind2]
    # # an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    # # un = index_su * an
    # # vn = index_sv * an

    # index_stu = stu[ind2]
    # index_stv = stv[ind2]
    # tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    # tun = index_stu * tn
    # tvn = index_stv * tn

    # angle = un * tun + vn * tvn + (an * tn)
    # index = [angle == 1.0]
    # angle[index] = 0.999
    # ang = np.arccos(angle)
    # mang = np.mean(ang)
    # mang = mang * 180 / np.pi
    # epe[ind2] = 0
    # mepe = np.mean(epe)

    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = spherical_coordinates.flow_warp_meshgrid(stu, stv)
        of_gt_endpoints_uv = spherical_coordinates.erp2spherical(of_gt_endpoints)

        of_eva_endpoints = spherical_coordinates.flow_warp_meshgrid(su, sv)
        of_eva_endpoints_uv = spherical_coordinates.erp2spherical(of_eva_endpoints)

        # get great circle distance
        epe = spherical_coordinates.great_circle_distance_uv(of_gt_endpoints_uv[0], of_gt_endpoints_uv[1], of_eva_endpoints_uv[0], of_eva_endpoints_uv[1])
    else:
        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)

    return epe


def RMSE(of_ground_truth, of_evaluation, spherical=False):
    """
    compute the root mean square error(RMSE) of optical flow

    retrun: rmse
    """
    # stu = of_ground_truth[:, :, 0]
    # stv = of_ground_truth[:, :, 1]
    # su = of_evaluation[:, :, 0]
    # sv = of_evaluation[:, :, 1]

    # # ignore the invalid data
    # idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    # stu[idxUnknow] = 0
    # stv[idxUnknow] = 0

    # ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)

    # diff_u = stu[ind2] - su[ind2]
    # diff_v = stv[ind2] - sv[ind2]

    # rmse = np.sqrt(np.sum(diff_u ** 2 + diff_v ** 2) / np.shape(diff_u)[0])
    rmse_mat = RMSE_mat(of_ground_truth, of_evaluation, spherical)
    rmse = np.sqrt(np.sum(rmse_mat) / (np.shape(rmse_mat)[0] * np.shape(rmse_mat)[1]))
    return rmse


def RMSE_mat(of_ground_truth, of_evaluation, spherical=False):
    """
    compute the root mean square error(RMSE) of optical flow.
    The return is the array with element-wise (u_e^i - u_g^i)^2 + (v_e^i - v_g^i)^2).

    retrun: a array
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # ignore the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0

    # ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    # diff_u = stu[ind2] - su[ind2]
    # diff_v = stv[ind2] - sv[ind2]

    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = spherical_coordinates.flow_warp_meshgrid(stu, stv)
        of_gt_endpoints_uv = spherical_coordinates.erp2spherical(of_gt_endpoints)

        of_eva_endpoints = spherical_coordinates.flow_warp_meshgrid(su, sv)
        of_eva_endpoints_uv = spherical_coordinates.erp2spherical(of_eva_endpoints)

        # get the Spherical Triangle angle
        rmse_mat = spherical_coordinates.great_circle_distance(of_gt_endpoints_uv, of_eva_endpoints_uv)
        rmse_mat = rmse_mat ** 2
    else:
        diff_u = stu - su
        diff_v = stv - sv
        # rmse_mat = np.sqrt(diff_u ** 2 + diff_v ** 2)
        rmse_mat = diff_u ** 2 + diff_v ** 2
        # rmse_mat = rmse_mat.reshape(np.shape(su))
    return rmse_mat


def AAE(of_ground_truth, of_evaluation, spherical=False):
    """
    The average angular error(AAE) 

    The result is between 0 and 2 * PI.
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # # remove the invalid data
    # idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    # stu[idxUnknow] = 0
    # stv[idxUnknow] = 0
    # su[idxUnknow] = 0
    # sv[idxUnknow] = 0
    # ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)

    # #
    # index_su = su#su[ind2]
    # index_sv = sv#sv[ind2]
    # index_stu = stu#stu[ind2]
    # index_stv = stv#stv[ind2]

    # valid_index = np.logical_or(ind2,  np.logical_not(idxUnknow))

    valid_index_gt = available_pixel(of_ground_truth)
    valid_index_eva = available_pixel(of_evaluation)
    valid_index = np.logical_and(valid_index_eva, valid_index_gt)

    # get the valid pixels index
    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = spherical_coordinates.flow_warp_meshgrid(stu, stv)
        of_gt_endpoints_uv = spherical_coordinates.erp2spherical(of_gt_endpoints)

        of_eva_endpoints = spherical_coordinates.flow_warp_meshgrid(su, sv)
        of_eva_endpoints_uv = spherical_coordinates.erp2spherical(of_eva_endpoints)

        of_origin_endpoints = spherical_coordinates.flow_warp_meshgrid(np.zeros(np.shape(su)), np.zeros(np.shape(su)))
        of_origin_endpoints_uv = spherical_coordinates.erp2spherical(of_origin_endpoints)

        # get the Spherical Triangle angle
        angles = spherical_coordinates.get_angle(of_origin_endpoints_uv, of_gt_endpoints_uv, of_eva_endpoints_uv)

    else:
        # compute the average angle
        uv_cross = su * stv - stu * sv
        uv_dot = su * stu + stv * sv
        angles = np.arctan2(uv_cross, uv_dot)
        angles = np.abs(angles)

    valid_pixel_number = np.count_nonzero(valid_index)
    aae = np.sum(angles[valid_index]) / valid_pixel_number
    return aae

    # x = sum(math.cos(a) for a in angles)
    # y = sum(math.sin(a) for a in angles)
    # if x == 0 and y == 0:
    #     raise ValueError("The angle average of the inputs is undefined: %r" % angles)
    # return math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi) * 180 / np.pi


def AAE_mat(of_ground_truth, of_evaluation, spherical=False):
    """
    Return the mat of the average angular error(AAE) 

    The result is between 0 and 2 * PI.
    """
    stu = of_ground_truth[:, :, 0]
    stv = of_ground_truth[:, :, 1]
    su = of_evaluation[:, :, 0]
    sv = of_evaluation[:, :, 1]

    # remove the invalid data
    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (np.absolute(stu) > SMALLFLOW) | (np.absolute(stv) > SMALLFLOW)
    index_su = su[ind2]
    index_sv = sv[ind2]

    index_stu = stu[ind2]
    index_stv = stv[ind2]

    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = spherical_coordinates.flow_warp_meshgrid(stu, stv)
        of_gt_endpoints_uv = spherical_coordinates.erp2spherical(of_gt_endpoints)

        of_eva_endpoints = spherical_coordinates.flow_warp_meshgrid(su, sv)
        of_eva_endpoints_uv = spherical_coordinates.erp2spherical(of_eva_endpoints)

        of_origin_endpoints = spherical_coordinates.flow_warp_meshgrid(np.zeros(np.shape(su)), np.zeros(np.shape(su)))
        of_origin_endpoints_uv = spherical_coordinates.erp2spherical(of_origin_endpoints)

        # get the Spherical Triangle angle
        angles_mat = spherical_coordinates.get_angle(of_origin_endpoints_uv, of_gt_endpoints_uv, of_eva_endpoints_uv)
    else:
        # compute the average angle
        uv_cross = index_su * index_stv - index_stu * index_sv
        uv_dot = index_su * index_stu + index_stv * index_sv
        angles = np.arctan2(uv_cross, uv_dot)
        angles_mat = angles.reshape(np.shape(su))

        angles_mat = abs(angles_mat)

    # x = sum(math.cos(a) for a in angles)
    # y = sum(math.sin(a) for a in angles)
    # if x == 0 and y == 0:
    #     raise ValueError(
    #         "The angle average of the inputs is undefined: %r" % angles)

    return angles_mat
