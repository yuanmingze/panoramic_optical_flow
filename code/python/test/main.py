import logging

import configuration as config
from replica360.configuration import ReplicaConfig

from utility import image_io
from utility import flow_estimate
from utility import flow_io
from utility import flow_vis
from utility import flow_evaluate
from utility import flow_postproc
from utility import fs_utility
from utility import image_evaluate
from utility import flow_warp
from utility import image_utility

import os
import csv
import sys
from shutil import copyfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt 
from skimage.transform import resize as ski_resize

from utility.logger import Logger
log = Logger(__name__)
log.logger.propagate = False


class OmniPhotoDataset():

    pano_dataset_root_dir = "D:/workdata/omniphoto_bmvc_2021/"
    # pano_dataset_root_dir = "/mnt/sda1/workdata/omniphoto_bmvc_2021/"
    result_output_dir = "D:/workspace_windows/panoramic_optical_flow/data/omniphoto_result/"
    pano_data_dir = "pano/"

    pano_output_dir = "result/"
    pano_output_csv = "result_omniphoto.csv"

    pano_image_height = 640

    # file name expression
    pano_rgb_image_filename_exp = "panoramic-{:04d}.jpg"
    pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    pano_opticalflow_forward_visual_filename_exp = "{:04d}_opticalflow_forward_pano_visual.jpg"
    pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"
    pano_opticalflow_backward_visual_filename_exp = "{:04d}_opticalflow_backward_pano_visual.jpg"

    # circle data
    dataset_circ_dirlist = [
        "Ballintoy",
        "BathAbbey2",
        "BathParadeGardens",
        "BeihaiPark",
        "Coast",
        "DublinShip1",
        "OsakaTemple6",
        "SecretGarden1",
        "Wulongting",
    ]


def of_estimate_omniphoto(omniphoto_dataset):
    """Get the our and optical flow result in replica. """
    dataset_dirlist = omniphoto_dataset.dataset_circ_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        print("processing the data folder {}".format(pano_image_folder))
        # input dir
        input_filepath = omniphoto_dataset.pano_dataset_root_dir + pano_image_folder + "/" + omniphoto_dataset.pano_data_dir + "/"
        # input index
        inputfile_list = fs_utility.dir_ls(input_filepath, ".jpg")
        pano_start_idx = 1
        pano_end_idx = len(inputfile_list) - 1

        # output folder
        output_pano_filepath = omniphoto_dataset.pano_dataset_root_dir + pano_image_folder + "/" + omniphoto_dataset.pano_output_dir
        output_dir = output_pano_filepath + "/our/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            pano_image_file_idx = int(inputfile_list[pano_image_idx][-8:-4])
            for forward_of in [True, False]:
                # 0) load image to CPU memory
                if forward_of:
                    tar_erp_image_filepath = inputfile_list[pano_image_idx + 1]
                    optical_flow_filepath = omniphoto_dataset.pano_opticalflow_forward_filename_exp.format(pano_image_file_idx)
                    optical_flow_vis_filepath = omniphoto_dataset.pano_opticalflow_forward_visual_filename_exp.format(pano_image_file_idx)
                else:
                    tar_erp_image_filepath = inputfile_list[pano_image_idx - 1]
                    optical_flow_filepath = omniphoto_dataset.pano_opticalflow_backward_filename_exp.format(pano_image_file_idx)
                    optical_flow_vis_filepath = omniphoto_dataset.pano_opticalflow_backward_visual_filename_exp.format(pano_image_file_idx)

                src_erp_image_filepath = inputfile_list[pano_image_idx]

                if pano_image_idx % 2 == 0:
                    print("Flow Method: {}, image folder: {}, srouce Image: {}, target image: {}, output flow file: {}".format(opticalflow_mathod, pano_image_folder, src_erp_image_filepath, tar_erp_image_filepath, optical_flow_filepath))
                src_erp_image = image_io.image_read(input_filepath + src_erp_image_filepath)
                tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filepath)

                # test and resize input image
                if src_erp_image.shape[0] * 2 != src_erp_image.shape[1]:
                    log.error("{} image size is {}".format(src_erp_image_filepath, src_erp_image.shape))
                if tar_erp_image.shape[0] * 2 != tar_erp_image.shape[1]:
                    log.error("{} image size is {}".format(tar_erp_image, src_erp_image.shape))
                if src_erp_image.shape[0] != omniphoto_dataset.pano_image_height:
                    src_erp_image = ski_resize(src_erp_image, (omniphoto_dataset.pano_image_height, omniphoto_dataset.pano_image_height * 2), anti_aliasing=True, preserve_range=True)
                    tar_erp_image = ski_resize(tar_erp_image, (omniphoto_dataset.pano_image_height, omniphoto_dataset.pano_image_height * 2), anti_aliasing=True, preserve_range=True)

                # 1) estimate optical flow
                flow_estimator = flow_estimate.PanoOpticalFlow()
                flow_estimator.debug_enable = False
                flow_estimator.debug_output_dir = None
                flow_estimator.padding_size_cubemap = padding_size
                flow_estimator.padding_size_ico = padding_size
                flow_estimator.flow2rotmat_method= "3D"
                flow_estimator.tangent_image_width_ico = 480
                optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                
                # 2) evaluate the optical flow and output result
                # output optical flow image
                result_opticalflow_filepath = output_dir + optical_flow_filepath
                flow_io.flow_write(optical_flow, result_opticalflow_filepath)
                optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)
                result_opticalflow_vis_filepath = output_dir + optical_flow_vis_filepath
                image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)


class ReplicaPanoDataset(ReplicaConfig):

    if sys.platform == 'win32':
        pano_dataset_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
        result_output_dir = "D:/workspace_windows/panoramic_optical_flow/data/replica_result/"
    elif sys.platform == 'linux':
        pano_dataset_root_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/"
        result_output_dir = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_result/"

    pano_data_dir = "pano/"

    pano_output_dir = "result/"
    pano_output_csv = "result_replica.csv"

    padding_size = None

    # circle data
    dataset_circ_dirlist = [
        "apartment_0_circ_1k_0",
        "apartment_1_circ_1k_0",
        # "apartment_2_circ_1k_0",
        "frl_apartment_0_circ_1k_0",
        # "frl_apartment_1_circ_1k_0",
        "frl_apartment_2_circ_1k_0",
        "frl_apartment_3_circ_1k_0",
        # "frl_apartment_4_circ_1k_0",
        # "frl_apartment_5_circ_1k_0",
        "hotel_0_circ_1k_0",
        "office_0_circ_1k_0",
        "office_1_circ_1k_0",
        "office_2_circ_1k_0",
        "office_3_circ_1k_0",
        "office_4_circ_1k_0",
        "room_0_circ_1k_0",
        "room_1_circ_1k_0",
        # "room_2_circ_1k_0",
    ]
    circle_start_idx = 4
    circle_end_idx = 6

    # line data
    dataset_line_dirlist = [
        "apartment_0_line_1k_0",
        "apartment_1_line_1k_0",
        # "apartment_2_line_1k_0",
        "frl_apartment_0_line_1k_0",
        # "frl_apartment_1_line_1k_0",
        "frl_apartment_2_line_1k_0",
        "frl_apartment_3_line_1k_0",
        # "frl_apartment_4_grid_1k_0",
        # "frl_apartment_5_grid_1k_0",
        "hotel_0_line_1k_0",
        "office_0_line_1k_0",
        "office_1_line_1k_0",
        "office_2_line_1k_0",
        "office_3_line_1k_0",
        "office_4_line_1k_0",
        "room_0_line_1k_0",
        "room_1_line_1k_0",
        # "room_2_grid_1k_0",
    ]
    line_start_idx = 4
    line_end_idx = 6

    # random data
    dataset_rand_dirlist = [
        "apartment_0_rand_1k_0",
        "apartment_1_rand_1k_0",
        # "apartment_2_rand_1k_0",
        "frl_apartment_0_rand_1k_0",
        # "frl_apartment_1_rand_1k_0",
        "frl_apartment_2_rand_1k_0",
        "frl_apartment_3_rand_1k_0",
        # "frl_apartment_4_rand_1k_0",
        # "frl_apartment_5_rand_1k_0",
        "hotel_0_rand_1k_0",
        "office_0_rand_1k_0",
        "office_1_rand_1k_0",
        "office_2_rand_1k_0",
        "office_3_rand_1k_0",
        "office_4_rand_1k_0",
        "room_0_rand_1k_0",
        "room_1_rand_1k_0",
        # "room_2_rand_1k_0",
    ]
    rand_start_idx = 4
    rand_end_idx = 6
    


def plot_padding_error(replica_dataset, padding_size_list, output_pdf_filepath = None, opticalflow_mathod = "our_weight"):
    """
    Plot the padding error.
    """
    x = []
    y = []
    for padding_size in padding_size_list:
        of_error_method_filepath = replica_dataset.pano_dataset_root_dir + opticalflow_mathod + "_" + str(padding_size) + "_" + replica_dataset.pano_output_csv

        x.append(padding_size)
        # load csv
        of_error_csv_file = open(of_error_method_filepath, "r")
        sepe = float(of_error_csv_file.readlines()[5].split(" ")[1])
        y.append(sepe)

    fig = plt.figure()
    margin_ratio = 0.04
    plt.plot(x, y, marker="o")
    plt.xlim(min(x) - max(x) * margin_ratio, max(x) + max(x)* margin_ratio)
    plt.ylim(min(y) - min(y) * margin_ratio, max(y) + max(y)* margin_ratio)

    # set trick label font size
    plt.xlabel("padding size", fontsize=22)
    plt.ylabel("SEPE", fontsize=22)

    # Set tick font size
    for label in (plt.xticks()[1] + plt.yticks()[1]):
        label.set_fontsize(18)

    # Set general font size
    plt.rcParams['font.size'] = '18'

    plt.show()
    if output_pdf_filepath is None:
        f.savefig(replica_dataset.pano_dataset_root_dir  + "padding.pdf", bbox_inches='tight')
    else:
        f.savefig(output_pdf_filepath, bbox_inches='tight')


def summary_error_dataset_replica(replica_dataset, opticalflow_mathod="our", dataset_dirlist = None, csv_postfix = None):
    """ Summary the error on whole replica. Collect all csv file's number. 
        # output the scv file to root of the whole dataset
    """
    # dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist
    row_counter = 0
    aae = 0
    epe = 0
    rms = 0
    aae_sph = 0
    epe_sph = 0
    rms_sph = 0

    # 1) iterate each scene's data
    for pano_image_folder in dataset_dirlist:
        # load the csv
        if replica_dataset.padding_size is None:
            of_error_csv_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "/" + replica_dataset.pano_output_csv
        else:
            of_error_csv_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "_" + str(replica_dataset.padding_size) + "/" + replica_dataset.pano_output_csv
        log.debug("read {}".format(of_error_csv_filepath))
        of_error_csv_file = open(of_error_csv_filepath, "r")
        of_error_csv = csv.DictReader(of_error_csv_file)
        for row in of_error_csv:
            aae += float(row["AAE"])
            epe += float(row["EPE"])
            rms += float(row["RMS"])
            aae_sph += float(row["SAAE"])
            epe_sph += float(row["SEPE"])
            rms_sph += float(row["SRMS"])
            row_counter += 1

        of_error_csv_file.close()

    # 2) output whole dataset summarized error information to file
    of_error_sum_csv_filepath = replica_dataset.pano_dataset_root_dir + "00_result_quantity_csv/"
    fs_utility.dir_make(of_error_sum_csv_filepath)

    if replica_dataset.padding_size is None:
        log.warn("The padding_size is None!")
        of_error_sum_csv_filepath = of_error_sum_csv_filepath + opticalflow_mathod + "_"
    else:
        of_error_sum_csv_filepath = of_error_sum_csv_filepath + opticalflow_mathod + "_" + str(replica_dataset.padding_size) + "_"

    if csv_postfix is not None:
        of_error_sum_csv_filepath = of_error_sum_csv_filepath + f"{csv_postfix}_"

    of_error_sum_csv_filepath = of_error_sum_csv_filepath + replica_dataset.pano_output_csv
    log.info("Ouput the datasets summary error to {}".format(of_error_sum_csv_filepath))
    log.info("output the error summary file to {}".format(of_error_sum_csv_filepath))
    msg = ""
    msg += "AAE: {}\n".format(aae / row_counter)
    msg += "EPE: {}\n".format(epe / row_counter)
    msg += "RMS: {}\n".format(rms / row_counter)
    msg += "AAE_SPH: {}\n".format(aae_sph / row_counter)
    msg += "EPE_SPH: {}\n".format(epe_sph / row_counter)
    msg += "RMS_SPH: {}\n".format(rms_sph / row_counter)
    msg += "\n===== Dataset & Optical flow method Information =====\n"
    from datetime import datetime
    msg += f"There are {row_counter} row data.\n"
    msg += "Evaluation Time: {}\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))	
    msg += f"Padding size: {replica_dataset.padding_size}\n"
    msg += f"Optical flow method: {opticalflow_mathod}\n"
    msg += "\nThe all datasets are:\n".format(rms_sph / row_counter)
    for dataset_name in dataset_dirlist:
        msg += f"\t{dataset_name}\n"
    file = open(of_error_sum_csv_filepath, "w")
    file.write(msg)
    file.close()


def summary_error_scene_replica(replica_dataset, opticalflow_mathod="our", dataset_dirlist = None, overwrite= False):
    """ Summary the error on replica for each scene."""

    # 1) iterate each scene's data
    for pano_image_folder in dataset_dirlist:
        # input dir
        of_gt_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir
        if replica_dataset.padding_size is None:
            of_eva_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "/"
        else:
            of_eva_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod +  "_" + str(replica_dataset.padding_size) + "/"
        if os.path.exists(of_eva_dir + replica_dataset.pano_output_csv) and not overwrite:
            log.warn("{} exist.".format(of_eva_dir + replica_dataset.pano_output_csv))
            continue
        log.info("Evaluate optical flow folder {}".format(of_eva_dir))
        flow_evaluate.opticalflow_metric_folder(of_eva_dir, of_gt_dir, mask_filename_exp=replica_dataset.replica_pano_mask_filename_exp,
                                                result_csv_filename=replica_dataset.pano_output_csv, visual_of_error=False, of_wraparound=True)


def warp_of_multi(scene_name, method_list, flow_filepath_exp, flo_filename,
                  rgb_filepath_exp, rgb_filename, rgb_gt_filename, output_dir):
    """Visualize optical flows error with same scale."""
    # load file from flo
    flow_data_list = []
    for method in method_list:
        flow_data_list.append(flow_io.flow_read(flow_filepath_exp.format(scene_name, method, flo_filename)))

    rgb_image_data = image_io.image_read(rgb_filepath_exp.format(scene_name, rgb_filename))
    rgb_gt_image_data = image_io.image_read(rgb_filepath_exp.format(scene_name, rgb_gt_filename))

    # if rgb_image_data.shape[0:2] != flow_data_list[0].shape[0:2]:
    #     print("The {} flow size is {}, the rgb image size is {}.".format(scene_name, flow_data_list[0].shape[0:2], rgb_image_data.shape[0:2]))
    #     rgb_image_data = image_io.image_resize(rgb_image_data, flow_data_list[0].shape[0:2])
    #     rgb_gt_image_data = image_io.image_resize(rgb_gt_image_data, flow_data_list[0].shape[0:2])

    # copy the source and target rgb image to target folder.
    rgb_filepath_src = rgb_filepath_exp.format(scene_name, rgb_filename)
    rgb_filepath_src_output = output_dir + "{}_{}".format(scene_name, rgb_filename)
    copyfile(rgb_filepath_src, rgb_filepath_src_output)
    rgb_filepath_tar = rgb_filepath_exp.format(scene_name, rgb_gt_filename)
    rgb_filepath_tar_output = output_dir + "{}_{}".format(scene_name, rgb_gt_filename)
    copyfile(rgb_filepath_tar, rgb_filepath_tar_output)

    # 0) warp source image with estimated optical flow
    counter = 0
    rgb_image_warp_data_list = []
    for flow_data in flow_data_list:
        rgb_image_data_temp = None
        if rgb_image_data.shape[0:2] != flow_data_list[counter].shape[0:2]:
            print("The {} flow size is {}, the rgb image size is {}.".format(scene_name, flow_data_list[counter].shape[0:2], rgb_image_data.shape[0:2]))
            rgb_image_data_temp = image_utility.image_resize(rgb_image_data, flow_data_list[counter].shape[0:2])
        else:
            rgb_image_data_temp = rgb_image_data

        warped_image_data = flow_warp.warp_backward(rgb_image_data_temp, flow_data)
        rgb_image_warp_data_list.append(warped_image_data)
        output_filepath = output_dir + "{}_{}_{}_wrap.jpg".format(scene_name, method_list[counter], flo_filename)
        image_io.image_save(warped_image_data, output_filepath)
        log.info("Output file to {}".format(output_filepath))
        counter += 1

    # 0) visualize the optical flow
    flow_min_ratio = 0.03
    flow_max_ratio = 0.85
    counter = 0
    for flow_data in flow_data_list:
        flow_vis_image = flow_vis.flow_to_color(flow_data, min_ratio=flow_min_ratio, max_ratio=flow_max_ratio)
        image_io.image_save(flow_vis_image, output_dir + "{}_{}_{}_flow_vis.jpg".format(scene_name, method_list[counter], flo_filename))
        counter += 1

    # 1) get the visualization error range
    counter = 0
    image_warp_error_list = []
    error_min_ratio = 0.1
    error_max_ratio = 0.90
    min_error = 9999999
    max_error = -1
    for rgb_image_warp_data in rgb_image_warp_data_list:
        rgb_gt_image_data_temp = None
        if rgb_image_warp_data.shape[0:2] != rgb_gt_image_data.shape[0:2]:
            print("The {} flow size is {}, the rgb image size is {}.".format(scene_name, flow_data_list[counter].shape[0:2], rgb_image_data.shape[0:2]))
            rgb_gt_image_data_temp = image_utility.image_resize(rgb_gt_image_data, rgb_image_warp_data.shape[0:2])
        else:
            rgb_gt_image_data_temp = rgb_gt_image_data

        image_warp_error_data = image_evaluate.diff_mat(rgb_gt_image_data_temp, rgb_image_warp_data)
        image_warp_error_list.append(image_warp_error_data)
        # max & min
        vmin_, vmax_ = image_evaluate.get_min_max(image_warp_error_data, error_min_ratio, error_max_ratio)
        if vmin_ < min_error:
            min_error = vmin_
        if vmax_ > max_error:
            max_error = vmax_
        counter += 1

    # 2) visualize the error  and output
    counter = 0
    for flow_error in image_warp_error_list:
        epe_mat_vis = flow_evaluate.error_visual(flow_error, max_error, min_error, bar_enable=False)
        # image_io.image_show(epe_mat_vis)
        image_io.image_save(epe_mat_vis, output_dir + "{}_{}_{}_warp_error_vis.jpg".format(scene_name, method_list[counter], flo_filename))
        counter += 1


def visualize_of_error_multi(flow_filepath_exp, flow_gt_filepath, scene_name, method_list, flo_filename, rgb_filename, mask_filename, output_dir):
    """Visualize optical flows error with same scale."""
    # mask the unavailable optical flow

    flow_data_gt = flow_io.flow_read(flow_gt_filepath.format(scene_name, flo_filename))

    of_mask = None
    mask_filename = flow_gt_filepath.format(scene_name, mask_filename)
    if mask_filename is not None and os.path.exists(mask_filename):
        of_mask = image_io.image_read(mask_filename)
        flow_data_gt = flow_evaluate.available_of(flow_data_gt, of_mask)

    image_height = flow_data_gt.shape[0]
    image_width = flow_data_gt.shape[1]

    # load file from flo
    flow_data_list = []
    for method in method_list:
        flow_data = flow_io.flow_read(flow_filepath_exp.format(scene_name, method, flo_filename))
        
        if flow_data.shape[:2] != flow_data_gt.shape[:2]:
            log.warn("The {} flow shape is {}, it's not match the ground truth size {}".format(method, flow_data.shape, flow_data_gt.shape))
            height_new = flow_data_gt.shape[0]
            width_new = flow_data_gt.shape[1]
            flow_data = flow_postproc.flow_resize(flow_data, width_new=width_new, height_new=height_new)
        flow_data = flow_evaluate.available_of(flow_data, of_mask)
        flow_data_list.append(flow_data)


    # 0) visualize the optical flow
    flow_min_ratio = 0.03
    flow_max_ratio = 0.85
    counter = 0
    for flow_data in flow_data_list:
        flow_vis_image = flow_vis.flow_to_color(flow_data, min_ratio=flow_min_ratio, max_ratio=flow_max_ratio)
        image_io.image_save(flow_vis_image, output_dir + "{}_{}_{}_flow_vis.jpg".format(scene_name, method_list[counter], flo_filename))
        counter += 1

    flow_vis_image = flow_vis.flow_to_color(flow_data_gt, min_ratio=flow_min_ratio, max_ratio=flow_max_ratio)
    visual_output_filepath = output_dir + "{}_{}_{}_flow_vis.jpg".format(scene_name, "gt", flo_filename)
    log.info("Output the optical flow visualization image to {}".format(visual_output_filepath))
    image_io.image_save(flow_vis_image, visual_output_filepath)

    # 1) get the visualization error range
    flow_error_list = []
    error_min_ratio = 0.05
    error_max_ratio = 0.85
    min_error = 9999999
    max_error = -1
    for flow_data in flow_data_list:
        flow_error_data, _ = flow_evaluate.EPE_mat(flow_data_gt, flow_data, spherical=True)
        flow_error_list.append(flow_error_data)
        # max & min
        vmin_, vmax_ = image_evaluate.get_min_max(flow_error_data, error_min_ratio, error_max_ratio)
        if vmin_ < min_error:
            min_error = vmin_
        if vmax_ > max_error:
            max_error = vmax_

    # 2) visualize the error and output
    counter = 0
    for flow_error in flow_error_list:
        flow_error[of_mask] = 0.0
        epe_mat_vis = flow_evaluate.error_visual(flow_error, max_error, min_error, bar_enable=False)
        # image_io.image_show(epe_mat_vis)
        image_io.image_save(epe_mat_vis, output_dir + "{}_{}_{}_error_vis.jpg".format(scene_name, method_list[counter], flo_filename))
        counter += 1

    # 3) create the bar
    epe_mat_vis = flow_evaluate.error_visual(flow_error_list[0], max_error, min_error, bar_enable=True)
    # image_io.image_show(epe_mat_vis)
    image_io.image_save(epe_mat_vis, output_dir + "{}_{}_{}_error_vis_bar.jpg".format(scene_name, method_list[0], flo_filename))

    # 4) the rgb image
    rgb_source_image = flow_gt_filepath.format(scene_name, rgb_filename)
    rgb_target_image = output_dir + "{}_{}".format(scene_name, rgb_filename)
    copyfile(rgb_source_image, rgb_target_image)


def visualize_of_dataset_replica(replica_dataset):
    """ Summary the error on replica for each scene."""
    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist
    min_ratio = 0.1
    max_ratio = 0.6

    error_map = False
    # 1) iterate each scene's data
    for pano_image_folder in dataset_dirlist:
        opticalflow_mathod_list = \
            fs_utility.dir_ls(replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir, None)
        for opticalflow_mathod in opticalflow_mathod_list:
            print("{} method {}".format(pano_image_folder, opticalflow_mathod))
            # input dir
            of_gt_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir
            of_eva_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "/"

            # output visualized optical flow
            opticalflow_filename_list = fs_utility.dir_ls(of_eva_dir, ".flo")
            for optical_flow_filename in opticalflow_filename_list:
                result_opticalflow_filepath = of_eva_dir + optical_flow_filename
                optical_flow = flow_io.flow_read(result_opticalflow_filepath)
                optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=min_ratio, max_ratio=max_ratio)
                result_opticalflow_vis_filepath = of_eva_dir + optical_flow_filename[:-4] + "_visual.jpg"
                image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)

            # output error image
            if error_map:
                flow_evaluate.opticalflow_metric_folder(of_eva_dir, of_gt_dir, mask_filename_exp=replica_dataset.replica_pano_mask_filename_exp,
                                                        result_csv_filename=replica_dataset.pano_output_csv, visual_of_error=False)


def of_estimate_replica_clean(dataset_dirlist, opticalflow_result_foldername=None, opticalflow_result_foldername_prefix = None):
    """Remove all result of specified method in the `result` folder.
    
    :param opticalflow_method_foldername: the folder prefix in the /dataset/result/ folder.
    :type: str
    """
    import re
    for pano_image_folder in dataset_dirlist:
        output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
        if opticalflow_result_foldername is not None:
            output_dir = output_pano_filepath + "/" + opticalflow_result_foldername + "/"
            fs_utility.dir_rm(output_dir)
            log.info("Remove folder: {}".format(output_dir))
        if opticalflow_result_foldername_prefix is not None:
            for opticalflow_result_foldername_method in os.listdir(output_pano_filepath):
                # if re.match(opticalflow_result_foldername_prefix + '_*[0-9.]*$', opticalflow_result_foldername_method):
                if re.match(opticalflow_result_foldername_prefix + '_*[0-9.a-zA-Z]*$', opticalflow_result_foldername_method):
                    folder_path = output_pano_filepath + opticalflow_result_foldername_method
                    fs_utility.dir_rm(folder_path)
                    log.info("Remove folder: {}".format(folder_path))


def of_estimate_replica(replica_dataset, opticalflow_mathod="our", dataset_dirlist = None, overwrite_exist = False):
    """Get the our and DIS's result in replica. """
    print("Optical flow method: {}, detaset_dirlist: {}, overwrite {}".format(opticalflow_mathod, dataset_dirlist, overwrite_exist))

    padding_size = replica_dataset.padding_size

    debug_output_dir = None

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        log.info("processing the data folder {} with padding {}".format(pano_image_folder, padding_size))
        # input dir
        input_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir + "/"
        # input index
        if pano_image_folder.find("line") != -1:
            pano_start_idx = replica_dataset.line_start_idx
            pano_end_idx = replica_dataset.line_end_idx
        elif pano_image_folder.find("circ") != -1:
            pano_start_idx = replica_dataset.circle_start_idx
            pano_end_idx = replica_dataset.circle_end_idx
        elif pano_image_folder.find("rand") != -1:
            pano_start_idx = replica_dataset.rand_start_idx
            pano_end_idx = replica_dataset.rand_end_idx
        else:
            log.error("{} folder naming is wrong".format(pano_image_folder))

        # output folder
        output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
        # the flo files output folder
        if padding_size is None:
            output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
        else:
            output_dir = output_pano_filepath + "/" + opticalflow_mathod + "_" + str(padding_size) + "/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        # estimate on the all images
        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            # forward and backward optical flow
            for forward_of in [True, False]:
                # 0) load image to CPU memory
                if forward_of:
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_forward_visual_filename_exp.format(pano_image_idx)
                else:
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_backward_visual_filename_exp.format(pano_image_idx)

                result_opticalflow_filepath = output_dir + optical_flow_filepath
                result_opticalflow_vis_filepath = output_dir + optical_flow_vis_filepath

                if os.path.exists(result_opticalflow_filepath) and not overwrite_exist:
                    log.info("{} exist, skip it.".format(result_opticalflow_filepath))
                    continue

                src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                if pano_image_idx % 2 == 0:
                    print("Flow Method: {}, image index: {}, srouce Image: {}, target image: {}, output flow file: {}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath, optical_flow_filepath))
                src_erp_image = image_io.image_read(input_filepath + src_erp_image_filepath)
                tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filepath)

                log.info("The padding size is {}".format(padding_size))

                flow_estimator = flow_estimate.PanoOpticalFlow()
                flow_estimator.debug_enable = False
                flow_estimator.debug_output_dir = None
                flow_estimator.padding_size_cubemap = padding_size
                flow_estimator.padding_size_ico = padding_size
                flow_estimator.flow2rotmat_method= "3D"
                flow_estimator.tangent_image_width_ico = 480

                # 1) estimate optical flow
                if opticalflow_mathod == "our":
                    # our method full term
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                elif opticalflow_mathod == "our_wo_weight":
                    # our method without blending method
                    flow_estimator.face_blending_method_ico = "straightforward"
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                elif opticalflow_mathod == "our_wo_erp":
                    # our method without erp warp
                    flow_estimator.erp_enable = False
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                elif opticalflow_mathod == "our_wo_cube":
                    # our method without cubemap warp
                    flow_estimator.cubemap_enable = False
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                elif opticalflow_mathod == "our_wo_ico":
                    # our method without icosahedron warp
                    flow_estimator.ico_enable = False
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                elif opticalflow_mathod == "dis":
                    # DIS optical flow
                    optical_flow = flow_estimate.of_methdod_DIS(src_erp_image, tar_erp_image)
                else:
                    log.error("the optical flow {} does not implement.")

                optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
                # 2) evaluate the optical flow and output result
                # output optical flow image
                flow_io.flow_write(optical_flow, result_opticalflow_filepath)
                optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)
                image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)


if __name__ == "__main__":


    padding_size = 0.4

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')
    parser.add_argument('--datatype',type=str, help='the dataset type [circ|line|rand|all]')
    parser.add_argument('--ofmethod',type=str, help='the dataset type [our, dis, raft, pwcnet, our_wo_weight, our_wo_cube, our_wo_ico]')

    args = parser.parse_args()
    
    task_list = []
    task_list.append(args.task)

    if -1 in task_list:
        exit()
        replica_dataset = ReplicaPanoDataset()
        # dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist + replica_dataset.dataset_rand_dirlist
        # dataset_dirlist = None
        dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist
        # opticalflow_mathod = "our_weight"
        # 0) remove folder with full folder name
        of_estimate_replica_clean(dataset_dirlist, opticalflow_result_foldername ="dis")
        # 1) remove folder with folder name prefix
        of_estimate_replica_clean(dataset_dirlist, opticalflow_result_foldername_prefix ="dis_")

    if 0 in task_list:
        # run our method
        # estimate optical flow for replica 360
        replica_pano_dataset = ReplicaPanoDataset()
        replica_pano_dataset.padding_size = padding_size

        # estimation the optical flow on the test datasets
        # our, dis, raft, pwcnet, our_wo_weight, our_wo_cube, our_wo_ico, our_wo_erp
        opticalflow_mathod = args.ofmethod
        ofmethod_list = ["our", "our_wo_weight", "our_wo_cube", "our_wo_ico", "our_wo_erp", "dis"]
        if not opticalflow_mathod in ofmethod_list:
            log.error("The optical flow method {} is not support.".format(opticalflow_mathod))
        if opticalflow_mathod == "dis":
             replica_pano_dataset.padding_size = None

        # dataset_dirlist = replica_pano_dataset.dataset_circ_dirlist + replica_pano_dataset.dataset_line_dirlist + replica_pano_dataset.dataset_rand_dirlist
        # # 0) create *.flo file for each image pair
        dataset_list = [args.datatype] #"line", "circ", "rand"
        if not dataset_list[0] in ['rand','line','circ', 'all']:
            log.error("Datatype {}, can not process".format(dataset_list))

        dataset_dirlist = []
        if "line" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_line_dirlist
        if "circ" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_circ_dirlist
        if "rand" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_rand_dirlist

        # 0) estimate optical flow
        of_estimate_replica(replica_pano_dataset, opticalflow_mathod, dataset_dirlist, overwrite_exist=False)
        # of_estimate_omniphoto(OmniPhotoDataset)

    if 1 in task_list:
        # summary error. for replica 360
        replica_pano_dataset = ReplicaPanoDataset()
        replica_pano_dataset.padding_size = padding_size

        # estimation the optical flow on the test datasets
        # our, dis, raft, pwcnet, our_wo_weight, our_wo_cube, our_wo_ico, our_wo_erp
        opticalflow_mathod = args.ofmethod
        ofmethod_list = ["our", "dis", "raft", "pwcnet", "omniflownet","our_wo_weight", "our_wo_cube", "our_wo_ico", "our_wo_erp"]
        if not opticalflow_mathod in ofmethod_list:
            log.error("The optical flow method {} is not support.".format(opticalflow_mathod))
        if opticalflow_mathod in ["dis", "raft", "pwcnet", "omniflownet"]:
            replica_pano_dataset.padding_size = None

        # dataset_dirlist = replica_pano_dataset.dataset_circ_dirlist + replica_pano_dataset.dataset_line_dirlist + replica_pano_dataset.dataset_rand_dirlist
        # # 0) create *.flo file for each image pair
        dataset_list = [args.datatype] #"line", "circ", "rand"
        if not dataset_list[0] in ['rand','line','circ', 'all']:
            log.error("Datatype {}, can not process".format(dataset_list))

        dataset_dirlist = []
        if "line" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_line_dirlist
        if "circ" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_circ_dirlist
        if "rand" in dataset_list or "all" in dataset_list:
            dataset_dirlist += replica_pano_dataset.dataset_rand_dirlist

        summary_all_dataset_csv_filename_postfix = args.datatype #rand,line,circ, all

        log.info("Optical flow method: {}, detaset_dirlist: {}".format(opticalflow_mathod, dataset_dirlist))

        # 1) evaluate error for each dataset folder (scene)
        summary_error_scene_replica(replica_pano_dataset, opticalflow_mathod, dataset_dirlist, overwrite=False)

        # 3) summary all dataset list error 
        summary_error_dataset_replica(replica_pano_dataset, opticalflow_mathod, dataset_dirlist, csv_postfix=summary_all_dataset_csv_filename_postfix)

        # visualize_of_dataset_replica(ReplicaPanoDataset)
    
    if 2 in task_list:
        # generate the error map and visualized optical flow for comparison
        replica_pano_dataset = ReplicaPanoDataset()
        # dataset_dirlist = replica_pano_dataset.dataset_circ_dirlist + replica_pano_dataset.dataset_line_dirlist
        dataset_dirlist = ["frl_apartment_0_rand_1k_0"]#["frl_apartment_0_circ_1k_0"] # apartment_0_circ_1k#apartment_0_circ_1k_0
        for scene_name in dataset_dirlist:
            # scene_name = "room_0_circ_1k_0"
            # method_list = ["our", "our_weight", "dis", "pwcnet", "raft", "omniflownet"]
            method_list = ["our_0.4", "dis", "pwcnet", "raft", "omniflownet"]
            flo_filename = "0004_opticalflow_backward_pano.flo"
            rgb_filename = "0004_rgb_pano.jpg"
            mask_filename = "0004_mask_pano.png"
            # mask_filename = None
            flow_filepath_exp = "D:/workdata/opticalflow_data_bmvc_2021/{}/result/{}/{}"
            flow_gt_filepath = "D:/workdata/opticalflow_data_bmvc_2021/{}/pano/{}"
            output_dir = replica_pano_dataset.result_output_dir
            # visualize the error heatmap
            visualize_of_error_multi(flow_filepath_exp, flow_gt_filepath, scene_name, method_list, flo_filename, rgb_filename,  mask_filename, output_dir=output_dir)
            # copy the original rgb images

    if 3 in task_list:
        # generate the warped rgb image for comparision
        method_list = ["our_weight", "dis", "pwcnet", "raft", "omniflownet"]
        # method_list = ["omniflownet"]
        output_dir = OmniPhotoDataset.result_output_dir
        fs_utility.dir_make(output_dir)
        # #
        scene_name = "BeihaiPark"
        rgb_filename = "panoramic-0550.jpg"
        rgb_gt_filename = "panoramic-0549.jpg"
        flo_filename = "0549_opticalflow_forward_pano.flo"
        # #
        # scene_name = "Ballintoy"
        # rgb_filename = "panoramic-0256.jpg"
        # rgb_gt_filename = "panoramic-0255.jpg"
        # flo_filename = "0255_opticalflow_forward_pano.flo"
        # #
        # scene_name = "BathAbbey2"
        # rgb_filename = "panoramic-0308.jpg"
        # rgb_gt_filename = "panoramic-0307.jpg"
        # flo_filename = "0307_opticalflow_forward_pano.flo"
        # #
        # scene_name = "DublinShip1"
        # rgb_filename = "panoramic-0548.jpg"
        # rgb_gt_filename = "panoramic-0547.jpg"
        # flo_filename = "0547_opticalflow_backward_pano.flo"
        # #
        # scene_name = "OsakaTemple6"
        # rgb_filename = "panoramic-0366.jpg"
        # rgb_gt_filename = "panoramic-0365.jpg"
        # flo_filename = "0365_opticalflow_backward_pano.flo"
        # #
        # scene_name = "SecretGarden1"
        # rgb_filename = "panoramic-0530.jpg"
        # rgb_gt_filename = "panoramic-0531.jpg"
        # flo_filename = "0531_opticalflow_backward_pano.flo"
        # #
        # scene_name = "Wulongting"
        # rgb_filename = "panoramic-0697.jpg"
        # rgb_gt_filename = "panoramic-0696.jpg"
        # flo_filename = "0696_opticalflow_forward_pano.flo"
        # #
        flow_filepath_exp = "D:/workdata/omniphoto_bmvc_2021/{}/result/{}/{}"
        rgb_filepath_exp = "D:/workdata/omniphoto_bmvc_2021/{}/pano/{}"
        warp_of_multi(scene_name, method_list, flow_filepath_exp, flo_filename,
                      rgb_filepath_exp, rgb_filename, rgb_gt_filename, output_dir=output_dir)

    if 4 in task_list:
        # ablation study for the padding size
        # padding_list = np.linspace(0.4, 0.8, num=11) # interval is 0.04
        padding_list = np.linspace(0.0, 0.8, num=21) # interval is 0.04
        print("The padding list is {}".format(padding_list))
        opticalflow_mathod = "our_weight"

        # # for padding_size in padding_list:
        # #     replicaPanoDataset = ReplicaPanoDataset
        # #     replicaPanoDataset.padding_size = padding_size
        # #     log.info("###  Test padding {}".format(padding_size))
        # #     of_estimate_replica(replicaPanoDataset, opticalflow_mathod)
        
        # # compute the optical flow with different padding size
        # from multiprocessing import Pool
        # params = []
        # for padding_size in padding_list:
        #     replicaPanoDataset = ReplicaPanoDataset()
        #     replicaPanoDataset.padding_size = padding_size
        #     params.append((replicaPanoDataset, opticalflow_mathod))
        # with Pool(processes=6) as pool:
        #     pool.starmap(of_estimate_replica, params)

        # # summary the error
        # for padding_size in padding_list:
        #     replicaPanoDataset = ReplicaPanoDataset()
        #     replicaPanoDataset.padding_size = padding_size
        #     summary_error_scene_replica(replicaPanoDataset, opticalflow_mathod)
        # for padding_size in padding_list:
        #     replicaPanoDataset = ReplicaPanoDataset()
        #     replicaPanoDataset.padding_size = padding_size
        #     summary_error_dataset_replica(replicaPanoDataset, opticalflow_mathod)

        output_pdf_filepath = None
        plot_padding_error(ReplicaPanoDataset, padding_list, output_pdf_filepath , opticalflow_mathod = "our_weight")
