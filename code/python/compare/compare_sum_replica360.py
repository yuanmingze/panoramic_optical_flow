import re
import os
import csv
import pathlib
from threading import Thread


import numpy as np

import flow_evaluate
import flow_io
import image_io
import flow_vis
import datasets_utility
import flow_postproc

"""
evalute the estimated optical flow and output the error to csv file
"""


def visual_of(root_dir, of_min=-300, of_max=300):
    """
    visual all flo files.
    """
    flow_file_str = r"[0-9]*_opticalflow_[a-zA-Z]*.flo"
    flow_file_re = re.compile(flow_file_str)
    of_list = []

    # load all optical flow file
    for item in pathlib.Path(root_dir).iterdir():
        if not flow_file_re.match(item.name) is None:
            of_list.append(item.name)

    # visualization optical flow
    for index in range(len(of_list)):
        item = of_list[index]
        of_data = flow_io.readFlowFile(root_dir + item)
        of_data_vis = flow_vis.flow_to_color(of_data, [of_min, of_max])
        of_vis_file_name = item.replace(".flo", ".jpg")
        image_io.image_save(of_data_vis, root_dir + of_vis_file_name)

        if index % 10 == 0:
            print("processing {}: {} to {}".format(index, item, of_vis_file_name))


def flow_evaluate_dataset(root_dir, flo_dir_list):
    """
    get the mean of all metirc.
    """
    from pathlib import Path
    dataset_csv_path = str(Path(flo_dir_list[0]).parent / "of_error.csv")

    # 0) load error data from each method result
    error_data_dict = {}

    for flo_path in flo_dir_list:
        # load data from csv
        csv_file = flo_path + "result.csv"
        flo_method = os.path.basename(os.path.dirname(flo_path))

        with open(csv_file) as f:
            print("{}:{}".format(flo_method, f.readline().split()))
            error_data_list = {"AAE": [], "EPE": [], "RME": []}
            for row_txt in f:
                row_data = row_txt.split(",")
                error_data_list["AAE"].append(float(row_data[2]))
                error_data_list["EPE"].append(float(row_data[3]))
                error_data_list["RME"].append(float(row_data[4]))

            error_data_dict[flo_method] = error_data_list

    # 1) comput mean error for each method
    error_csv_file = open(dataset_csv_path, 'w', newline='')
    output_csv_header = "# method, AAE, EPE, RME\n"
    error_csv_file.write(output_csv_header)

    for method in error_data_dict.keys():
        error_csv_file.write("{}".format(method))
        for error_type in error_data_dict[method]:
            error_data = error_data_dict[method][error_type]
            error_data_np = np.asarray(error_data)
            error_csv_file.write(",{}".format(np.mean(error_data_np)))
        error_csv_file.write("\n")
    error_csv_file.close()


def flow_evaluate_frames(flow_gt_dir, flow_estimated_dir):
    """
    evaluate the quality of estimated optical flow.
    """
    # get data list
    min_index, max_index, image_list, of_forward_list, of_backward_list = datasets_utility.scene_of_folder(flow_gt_dir)
    _, _, image_list, of_forward_list, of_backward_list = datasets_utility.scene_of_folder(flow_gt_dir)

    # estimate and output to csv file
    output_csv_header = "# index, file_name, AAE, EPE, RME, AAE_SC, EPE_SC, RME_SC\n"
    warped_csv_file_path = flow_estimated_dir + "result.csv"
    error_csv_file = open(warped_csv_file_path, 'w', newline='')
    error_csv = csv.writer(error_csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    error_csv_file.write(output_csv_header)

    for index in range(min_index, max_index + 1):
        if index % 10 == 0:
            print("evaluate optical flow {}: {}".format(os.path.basename(os.path.dirname(flow_estimated_dir)), of_forward_list[index]))

        for forward_of in [0, 1]:
            of_gt_file_path = None
            flo_filename = None
            of_estimated_file_path = None
            if forward_of == 0:
                of_gt_file_path = flow_gt_dir + of_backward_list[index]
                of_estimated_file_path = flow_estimated_dir + of_backward_list[index]
                flo_filename = of_backward_list[index]
            else:
                of_gt_file_path = flow_gt_dir + of_forward_list[index]
                of_estimated_file_path = flow_estimated_dir + of_forward_list[index]
                flo_filename = of_forward_list[index]

            try:
                of_gt = flow_io.readFlowFile(of_gt_file_path)
            except KeyError:
                print("optical flow file do not exist {}".format(of_gt_file_path))
                continue

            # process the warp around
            of_gt = flow_postproc.convert_warp_around(of_gt)

            of_estimated = flow_io.readFlowFile(of_estimated_file_path)

            # 1) output the error to CSV file
            # 1-0) comput the error in ERP image
            row = [index * 2 + forward_of, of_gt_file_path]
            row.append(flow_evaluate.AAE(of_gt, of_estimated))
            row.append(flow_evaluate.EPE(of_gt, of_estimated))
            row.append(flow_evaluate.RMSE(of_gt, of_estimated))
            # 1-1) compute the error in spherical coordinates
            row.append(flow_evaluate.AAE(of_gt, of_estimated, spherical=True))
            row.append(flow_evaluate.EPE(of_gt, of_estimated, spherical=True))
            row.append(flow_evaluate.RMSE(of_gt, of_estimated, spherical=True))
            error_csv.writerow(row)

            # 2) output the error image to jpg files
            flow_error_vis(of_gt, of_estimated, flow_estimated_dir, flo_filename)
 

    error_csv_file.close()


def flow_error_vis(of_gt, of_estimated, flo_file_root_dir, flo_filename):
    """ Visualize the optical flow error and save to image file.

    :param of_gt: The optical flow ground truth data.
    :type of_gt: numpy
    :param of_estimated: The estimated optical flow data.
    :type of_estimated: numpy
    :param flo_file_root_dir: The flo file root folder.
    :type flo_file_root_dir: str
    :param flo_filename: The flo file name.
    :type flo_filename: str
    """    
    # of_estimated_file_path = None
    epe_error_vis = True
    aae_error_vis = True
    rmse_error_vis = True
    aae_error_cs_vis = True
    rmse_error_cs_vis = True
    epe_error_cs_vis = True

    # 2-1) comput the error in ERP image
    if aae_error_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_aae.jpg"
        aae_data = flow_evaluate.AAE_mat(of_gt, of_estimated)
        aae_data_visual = flow_evaluate.error_visual(aae_data)
        image_io.image_save(aae_data_visual, error_image_path)

    if epe_error_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_epe.jpg"
        epe_data = flow_evaluate.EPE_mat(of_gt, of_estimated)
        epe_data_visual = flow_evaluate.error_visual(epe_data)
        image_io.image_save(epe_data_visual, error_image_path)

    if rmse_error_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_rmse.jpg"
        rmse_data = flow_evaluate.RMSE_mat(of_gt, of_estimated)
        rmse_data_visual = flow_evaluate.error_visual(rmse_data)
        image_io.image_save(aae_data_visual, error_image_path)

    # 2-2) compute the error in spherical coordinates
    if aae_error_cs_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_aae_spherical.jpg"
        aae_data = flow_evaluate.AAE_mat(of_gt, of_estimated, spherical=True)
        aae_data_visual = flow_evaluate.error_visual(aae_data)
        image_io.image_save(aae_data_visual, error_image_path)

    if epe_error_cs_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_epe_spherical.jpg"
        epe_data = flow_evaluate.EPE_mat(of_gt, of_estimated, spherical=True)
        epe_data_visual = flow_evaluate.error_visual(epe_data, verbose=True)
        image_io.image_save(epe_data_visual, error_image_path)

    if rmse_error_cs_vis:
        # of_estimated_error_file_name = os.path.splitext(of_estimated_file_path)[0]
        error_image_path = flo_file_root_dir + flo_filename + "_rmse_spherical.jpg"
        rmse_data = flow_evaluate.RMSE_mat(of_gt, of_estimated, spherical=True)
        rmse_data_visual = flow_evaluate.error_visual(rmse_data)
        image_io.image_save(rmse_data_visual, error_image_path)


if __name__ == "__main__":
    # root folder of replica 360 output folder
    dataset_root = "/mnt/sda1/workdata/opticalflow_data/replica_360/"
    dataset_dir_list = ["apartment_0",  "hotel_0",  "office_0",  "office_4",  "room_0",  "room_1"]

    for dataset_dir in dataset_dir_list:
        print("========{}==========".format(dataset_dir))

        root_dir = dataset_root + dataset_dir + "/"

        replica_gt_dir = root_dir + "replica_seq_data/"
        flo_dir_dis = root_dir + "dis/"
        flo_dir_flownet2 = root_dir + "flownet2/"
        flo_dir_pwcnet = root_dir + "pwcnet/"
        flo_dir_raft = root_dir + "raft/"
        flo_dir_list = [flo_dir_dis, flo_dir_flownet2, flo_dir_pwcnet, flo_dir_raft]

        # flow_evaluate_frames(replica_gt_dir, flo_dir_dis)

        # 0) visual all output flo files
        print("--1) Visualization.")
        for dataset_dir in flo_dir_list:
            print("Visualization {}".format(os.path.basename(os.path.dirname(dataset_dir))))
            visual_of(dataset_dir)

        # 1) get the error csv file
        print("--2) Evaluate the optical flow error.")
        thread_list = []
        for dataset_dir in flo_dir_list:
            print("Evaluate {}".format(os.path.basename(os.path.dirname(dataset_dir))))
            # flow_evaluate_frames(replica_gt_dir, flo_dir_dis)
            flo_thread = Thread(target=flow_evaluate_frames, args=(replica_gt_dir, dataset_dir))
            thread_list.append(flo_thread)
            flo_thread.start()

        for of_thread in thread_list:
            of_thread.join()

        # 2)  get the mean error for a dataset
        print("--3) Summary the result.")
        flow_evaluate_dataset(root_dir, flo_dir_list)
