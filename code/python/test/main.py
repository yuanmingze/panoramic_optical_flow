import configuration as config

from utility import image_io
from utility import flow_estimate
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import flow_evaluate
from utility import flow_postproc
from utility import fs_utility

import csv

from utility import flow_warp
from skimage.transform import resize as ski_resize

from pathlib import Path
from replica360.configuration import ReplicaConfig

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


class OmniPhotoDataset():

    pano_dataset_root_dir = "D:/workdata/omniphoto_bmvc_2021/"
    # pano_dataset_root_dir = "/mnt/sda1/workdata/omniphoto_bmvc_2021/"
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
        # "Ballintoy",
        # "BathAbbey2",
        # "BathParadeGardens",
        # "BeihaiPark",
        # "Coast",
        # "DublinShip1",
        "OsakaTemple6",
        # "SecretGarden1",
        # "Wulongting",
    ]


def of_estimate_omniphoto(omniphoto_dataset, opticalflow_mathod="our"):
    """Get the our and optical flow result in replica. """
    dataset_dirlist = omniphoto_dataset.dataset_circ_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        print("processing the data folder {}".format(pano_image_folder))
        # input dir
        input_filepath = omniphoto_dataset.pano_dataset_root_dir + pano_image_folder + "/" + omniphoto_dataset.pano_data_dir + "/"
        # input index
        inputfile_list = fs_utility.dir_grep(input_filepath, ".jpg")
        pano_start_idx = 1
        pano_end_idx = len(inputfile_list) - 1

        # output folder
        output_pano_filepath = omniphoto_dataset.pano_dataset_root_dir + pano_image_folder + "/" + omniphoto_dataset.pano_output_dir
        output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
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
                    print("{} Flow Method: {}\n{}\n{}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))
                src_erp_image = image_io.image_read(input_filepath + src_erp_image_filepath)
                tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filepath)

                # test and resize input image
                if src_erp_image.shape[0] * 2 != src_erp_image.shape[1]:
                    log.error("{} image size is {}".fromat(src_erp_image_filepath, src_erp_image.shape))
                if tar_erp_image.shape[0] * 2 != tar_erp_image.shape[1]:
                    log.error("{} image size is {}".fromat(tar_erp_image, src_erp_image.shape))
                if src_erp_image.shape[0] != omniphoto_dataset.pano_image_height:
                    src_erp_image = ski_resize(src_erp_image, (omniphoto_dataset.pano_image_height, omniphoto_dataset.pano_image_height * 2), anti_aliasing=True, preserve_range=True)
                    tar_erp_image = ski_resize(tar_erp_image, (omniphoto_dataset.pano_image_height, omniphoto_dataset.pano_image_height * 2), anti_aliasing=True, preserve_range=True)

                # 1) estimate optical flow
                if opticalflow_mathod == "our":
                    optical_flow = flow_estimate.pano_of_0(src_erp_image, tar_erp_image, debug_output_dir=None)
                elif opticalflow_mathod == "dis":
                    optical_flow = flow_estimate.of_methdod_DIS(src_erp_image, tar_erp_image)
                else:
                    log.error("the optical flow {} does not implement.")

                optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
                # 2) evaluate the optical flow and output result
                # output optical flow image
                result_opticalflow_filepath = output_dir + optical_flow_filepath
                flow_io.flow_write(optical_flow, result_opticalflow_filepath)
                optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)
                result_opticalflow_vis_filepath = output_dir + optical_flow_vis_filepath
                image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)


class ReplicaPanoDataset(ReplicaConfig):

    # pano_dataset_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
    pano_dataset_root_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/"
    pano_data_dir = "pano/"

    pano_output_dir = "result/"
    pano_output_csv = "result_replica.csv"

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
    circle_start_idx = 1
    circle_end_idx = 35

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
    line_start_idx = 2
    line_end_idx = 8


def summary_error_dataset_replica(replica_dataset, opticalflow_mathod="our"):
    """ Summary the error on whole replica. Collect all csv file's number. """
    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist
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
        of_error_csv_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "/" + replica_dataset.pano_output_csv
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

    # 2) output whole information to file
    of_error_method_filepath = replica_dataset.pano_dataset_root_dir + opticalflow_mathod + "_" + replica_dataset.pano_output_csv
    log.info("output the error summary file to {}".format(of_error_method_filepath))
    msg = ""
    msg += f"There are {row_counter} row data.\n"
    msg += "AAE: {}\n".format(aae / row_counter)
    msg += "EPE: {}\n".format(epe / row_counter)
    msg += "RMS: {}\n".format(rms / row_counter)
    msg += "AAE_SPH: {}\n".format(aae_sph / row_counter)
    msg += "EPE_SPH: {}\n".format(epe_sph / row_counter)
    msg += "RMS_SPH: {}\n".format(rms_sph / row_counter)
    file = open(of_error_method_filepath, "w")
    file.write(msg)
    file.close()


def summary_error_scene_replica(replica_dataset, opticalflow_mathod="our"):
    """ Summary the error on replica for each scene."""
    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist

    # 1) iterate each scene's data
    for pano_image_folder in dataset_dirlist:
        # input dir
        of_gt_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir
        of_eva_dir = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir + opticalflow_mathod + "/"
        flow_evaluate.opticalflow_metric_folder(of_eva_dir, of_gt_dir, mask_filename_exp=replica_dataset.replica_pano_mask_filename_exp,
                                                result_csv_filename=replica_dataset.pano_output_csv,  visual_of_error=False)


def of_estimate_replica(replica_dataset, opticalflow_mathod="our"):
    """Get the our and DIS's result in replica. """
    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        print("processing the data folder {}".format(pano_image_folder))
        # input dir
        input_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir + "/"
        # input index
        if pano_image_folder.find("line") != -1:
            pano_start_idx = replica_dataset.line_start_idx
            pano_end_idx = replica_dataset.line_end_idx
        elif pano_image_folder.find("circ") != -1:
            pano_start_idx = replica_dataset.circle_start_idx
            pano_end_idx = replica_dataset.circle_end_idx
        else:
            log.error("{} folder naming is wrong".format(pano_image_folder))

        # output folder
        output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
        output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            for forward_of in [True, False]:
                # 0) load image to CPU memory
                if forward_of:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_forward_visual_filename_exp.format(pano_image_idx)
                else:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_backward_visual_filename_exp.format(pano_image_idx)

                mask_erp_image_filepath = replica_dataset.replica_pano_mask_filename_exp.format(pano_image_idx)

                if pano_image_idx % 2 == 0:
                    print("{} Flow Method: {}\n{}\n{}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))
                src_erp_image = image_io.image_read(input_filepath + src_erp_image_filepath)
                tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filepath)

                # 1) estimate optical flow
                if opticalflow_mathod == "our":
                    optical_flow = flow_estimate.pano_of_0(src_erp_image, tar_erp_image, debug_output_dir=None)
                elif opticalflow_mathod == "dis":
                    optical_flow = flow_estimate.of_methdod_DIS(src_erp_image, tar_erp_image)
                else:
                    log.error("the optical flow {} does not implement.")

                optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
                # 2) evaluate the optical flow and output result
                # output optical flow image
                result_opticalflow_filepath = output_dir + optical_flow_filepath
                flow_io.flow_write(optical_flow, result_opticalflow_filepath)
                optical_flow_vis = flow_vis.flow_to_color(optical_flow)
                result_opticalflow_vis_filepath = output_dir + optical_flow_vis_filepath
                image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)


if __name__ == "__main__":
    opticalflow_mathod = "our"  # our, dis, raft, pwcnet
    # of_estimate_replica(ReplicaPanoDataset, opticalflow_mathod)
    # summary_error_scene_replica(ReplicaPanoDataset, opticalflow_mathod)
    # summary_error_dataset_replica(ReplicaPanoDataset, opticalflow_mathod)
    of_estimate_omniphoto(OmniPhotoDataset, opticalflow_mathod)
