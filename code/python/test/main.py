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

from pathlib import Path
from replica360.configuration import ReplicaConfig

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


class ReplicaPanoDataset(ReplicaConfig):

    pano_dataset_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
    pano_data_dir = "pano"

    pano_output_dir = "result"
    pano_output_csv = "result.csv"

    # circle data
    dataset_circ_dirlist = [
        "apartment_0_circ_1k_0",
        # "apartment_1_circ_1k_0",
        # "apartment_2_circ_1k_0",
        # "frl_apartment_0_circ_1k_0",
        # "frl_apartment_1_circ_1k_0",
        # "frl_apartment_2_circ_1k_0",
        # "frl_apartment_3_circ_1k_0",
        # "frl_apartment_4_circ_1k_0",
        # "frl_apartment_5_circ_1k_0",
        # "hotel_0_circ_1k_0",
        # "office_0_circ_1k_0",
        # "office_1_circ_1k_0",
        # "office_2_circ_1k_0",
        # "office_3_circ_1k_0",
        # "office_4_circ_1k_0",
        # "room_0_circ_1k_0",
        # "room_1_circ_1k_0",
        # "room_2_circ_1k_0",
    ]
    circle_start_idx = 3
    circle_end_idx = 20

    # line data
    dataset_line_dirlist = [
        # "apartment_0_line_1k_0",
        # "apartment_1_line_1k_0",
        # "apartment_2_line_1k_0",
        # "frl_apartment_0_line_1k_0",
        # "frl_apartment_1_line_1k_0",
        # "frl_apartment_2_line_1k_0",
        # "frl_apartment_3_line_1k_0",
        # "frl_apartment_4_grid_1k_0",
        # "frl_apartment_5_grid_1k_0",
        # "hotel_0_line_1k_0",
        # "office_0_line_1k_0",
        # "office_1_line_1k_0",
        # "office_2_line_1k_0",
        # "office_3_line_1k_0",
        # "office_4_line_1k_0",
        # "room_0_line_1k_0",
        # "room_1_line_1k_0",
        # "room_2_grid_1k_0",
    ]
    line_start_idx = 1
    line_end_idx = 9


def replica_test(replica_dataset):

    opticalflow_mathod = "dis"

    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        print("processing the data folder {}".format(pano_image_folder))
        # input dir
        input_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir + "/"
        # input index
        if pano_image_folder.find("line") is not None:
            pano_start_idx = replica_dataset.line_start_idx
            pano_end_idx = replica_dataset.line_end_idx   
        elif pano_image_folder.find("circ") is not None:
            pano_start_idx = replica_dataset.circle_start_idx
            pano_end_idx = replica_dataset.circle_end_idx
        else:
            log.error("{} folder naming is wrong".format(pano_image_folder))   

        # output folder
        output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
        output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        # result csv file
        result_csv_filepath = output_dir + replica_dataset.pano_output_csv
        result_file = open(result_csv_filepath, mode='w')
        result_csv_file = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_csv_file.writerow(["src_filepath", "tar_filepath", "AAE", 'EPE', 'RMS', "SAAE", "SEPE", "SRMS"])

        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            for forward_of in [True, False]:


                # 0) load image to CPU memory
                if forward_of:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                    mask_erp_image_filepath = replica_dataset.replica_pano_mask_filename_exp.format(pano_image_idx)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_forward_visual_filename_exp.format(pano_image_idx)
                else:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                    mask_erp_image_filepath = replica_dataset.replica_pano_mask_filename_exp.format(pano_image_idx)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = replica_dataset.replica_pano_opticalflow_backward_visual_filename_exp.format(pano_image_idx)

                if pano_image_idx % 2 == 0:
                    print("{}\n{}\n{}".format(pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))
                src_erp_image = image_io.image_read(input_filepath + src_erp_image_filepath)
                tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filepath)
                mask_erp_image = image_io.image_read(input_filepath + mask_erp_image_filepath)
                optical_flow_gt = flow_io.read_flow_flo(input_filepath + optical_flow_filepath)

                # 1) estimate optical flow
                if opticalflow_mathod == "our":
                    optical_flow = flow_estimate.pano_of_0(src_erp_image, tar_erp_image, debug_output_dir=None)
                elif opticalflow_mathod == "raft":
                    optical_flow = flow_estimate.raft(src_erp_image, tar_erp_image, debug_output_dir=None)
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

                # output optical flow metric
                aae = flow_evaluate.AAE(optical_flow_gt, optical_flow, spherical=False, of_mask=mask_erp_image)
                aae_sph = flow_evaluate.AAE(optical_flow_gt, optical_flow, spherical=True, of_mask=mask_erp_image)
                epe = flow_evaluate.EPE(optical_flow_gt, optical_flow, spherical=False, of_mask=mask_erp_image)
                epe_sph = flow_evaluate.EPE(optical_flow_gt, optical_flow, spherical=True, of_mask=mask_erp_image)
                rms = flow_evaluate.RMSE(optical_flow_gt, optical_flow, spherical=False, of_mask=mask_erp_image)
                rms_sph = flow_evaluate.RMSE(optical_flow_gt, optical_flow, spherical=True, of_mask=mask_erp_image)
                result_csv_file.writerow([src_erp_image_filepath, tar_erp_image_filepath, aae, epe, rms, aae_sph, epe_sph, rms_sph])
                result_file.flush()

        result_csv_file.close()


if __name__ == "__main__":
    replica_test(ReplicaPanoDataset)
