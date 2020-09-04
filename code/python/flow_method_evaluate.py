import re
import os
import sys
import csv
import pathlib

import utility.flow_estimate
import utility.flow_evaluate
from utility import flow_io
from utility import image_io
from utility import flow_vis

"""
Evaluate the quality of optical flow in panoramic images optical flow.
"""

INDEX_MAX = sys.maxsize


def scene_folder(root_dir):
    """
    """
    # search all rgb image
    index_digit_re = r"\d{4}"
    index_digit_format = r"{:04d}"

    flow_fw_fne_str = index_digit_re + r"_opticalflow_forward.flo"
    flow_bw_fne_str = index_digit_re + r"_opticalflow_backward.flo"

    rgb_image_fne = re.compile(index_digit_re + r"_rgb.jpg")
    flow_fw_fne = re.compile(flow_fw_fne_str)
    flow_bw_fne = re.compile(flow_bw_fne_str)

    # scan all file to get the index, and load image and of list
    of_forward_list = {}
    of_backward_list = {}
    image_list = {}
    min_index = INDEX_MAX
    max_index = -1

    for item in pathlib.Path(root_dir).iterdir():
        index_number = re.search(index_digit_re, item.name).group()
        if index_number is None:
            continue

        index_number = int(index_number)
        if index_number > max_index:
            max_index = index_number
        if index_number < min_index:
            min_index = index_number

        if not rgb_image_fne.match(item.name) is None:
            image_list[index_number] = item.name
        elif not flow_fw_fne.match(item.name) is None:
            of_forward_list[index_number] = item.name
        elif not flow_bw_fne.match(item.name) is None:
            of_backward_list[index_number] = item.name

    return min_index, max_index, image_list, of_forward_list, of_backward_list


def flow_estimate(root_dir, of_method):
    """
    estimate optical flow with specified method.
    """
    output_dir = root_dir + "{}/".format(of_method)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_dir = root_dir + "replica_seq_data/"

    # search all rgb image
    index_digit_re = r"\d{4}"
    index_digit_format = r"{:04d}"
    rgb_image_fne = re.compile(index_digit_re + r"_rgb.jpg")

    flow_fw_fne_str = index_digit_re + r"_opticalflow_forward.flo"
    flow_bw_fne_str = index_digit_re + r"_opticalflow_backward.flo"

    flow_fw_fne = re.compile(flow_fw_fne_str)
    flow_bw_fne = re.compile(flow_bw_fne_str)

    image_list = []
    min_index = INDEX_MAX
    max_index = -1
    for item in pathlib.Path(input_dir).iterdir():
        index_number = re.search(index_digit_re, item.name)
        if index_number is None:
            continue

        index_number = int(index_number.group())
        if index_number > max_index:
            max_index = index_number
        if index_number < min_index:
            min_index = index_number

        if not rgb_image_fne.match(item.name) is None:
            image_list.append(item.name)

    image_list.sort()
    frame_number = len(image_list)

    # compute the optical flow A --> B and B --> A
    for index in range(min_index, max_index + 1):
        rgb_image_A_file_name = image_list[(index - 1 + frame_number) % frame_number]
        rgb_image_B_file_name = image_list[index]

        if index % 1 == 0:
            print("compute optical flow {} to {}".format(rgb_image_A_file_name, rgb_image_B_file_name))
            print("compute optical flow {} to {}".format(rgb_image_B_file_name, rgb_image_A_file_name))

        rgb_image_A_file_name = input_dir + rgb_image_A_file_name
        rgb_image_B_file_name = input_dir + rgb_image_B_file_name

        rgb_image_A_data = image_io.image_read(rgb_image_A_file_name)
        rgb_image_B_data = image_io.image_read(rgb_image_B_file_name)

        #
        flow_A_B_data = utility.flow_estimate.DIS(rgb_image_A_data, rgb_image_B_data)
        flow_A_B_file_name = flow_fw_fne_str.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_A_B_data, output_dir + flow_A_B_file_name)

        flow_visual_data = flow_vis.flow_to_color(flow_A_B_data)
        image_io.image_save(flow_visual_data, output_dir + os.path.splitext(flow_A_B_file_name)[0] + "_vis.jpg")

        flow_B_A_data = utility.flow_estimate.DIS(rgb_image_B_data, rgb_image_A_data)
        flow_B_A_file_name = flow_bw_fne_str.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_B_A_data, output_dir + flow_B_A_file_name)

        flow_visual_data = flow_vis.flow_to_color(flow_B_A_data)
        image_io.image_save(flow_visual_data, output_dir + os.path.splitext(flow_B_A_file_name)[0] + "_vis.jpg")


def flow_evaluate(root_dir, method_name):
    """
    evaluate the quality of estimated optical flow.
    """
    output_folder = root_dir + method_name + "/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # get data list
    flow_gt_dir = root_dir + "replica_seq_data/"
    min_index, max_index, image_list, of_forward_list, of_backward_list = scene_folder(flow_gt_dir)

    flow_estimated_dir = root_dir + method_name + "/"
    _, _, image_list, of_forward_list, of_backward_list = scene_folder(flow_gt_dir)

    # estimate and output to csv file
    output_csv_header = "# index, file_name, AAE, EPE, RME"
    warped_csv_file_path = output_folder + "result.csv"
    error_csv_file = open(warped_csv_file_path, 'w', newline='')
    error_csv = csv.writer(error_csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    error_csv.writerow(output_csv_header)

    for index in range(min_index, max_index + 1):
        print("evaluate optical flow {}".format(of_forward_list[index]))
        try:
            of_fw_gt_file_path = flow_gt_dir + of_forward_list[index]
            of_fw_estimated_file_path = flow_estimated_dir + of_forward_list[index]
        except KeyError:
            print("optical flow file do not exist {}".format(of_fw_gt_file_path))
            continue

        of_gt = flow_io.readFlowFile(of_fw_gt_file_path)
        of_estimated = flow_io.readFlowFile(of_fw_estimated_file_path)
        row = [index * 2, of_fw_gt_file_path]
        row.append(utility.flow_evaluate.AAE(of_gt, of_estimated))
        row.append(utility.flow_evaluate.EPE(of_gt, of_estimated))
        row.append(utility.flow_evaluate.RMSE(of_gt, of_estimated))
        error_csv.writerow(row)

        try:
            of_bw_gt_file_path = flow_gt_dir + of_backward_list[index]
            of_bw_estimated_file_path = flow_estimated_dir + of_backward_list[index]
        except KeyError:
            print("optical flow file do not exist {}".format(of_bw_gt_file_path))
            continue

        of_gt = flow_io.readFlowFile(of_bw_gt_file_path)
        of_estimated = flow_io.readFlowFile(of_bw_estimated_file_path)
        row = [index * 2 + 1, of_bw_gt_file_path]
        row.append(utility.flow_evaluate.AAE(of_gt, of_estimated))
        row.append(utility.flow_evaluate.EPE(of_gt, of_estimated))
        row.append(utility.flow_evaluate.RMSE(of_gt, of_estimated))
        error_csv.writerow(row)

        print("evaluate optical flow {}".format(of_backward_list[index]))

    error_csv_file.close()


if __name__ == "__main__":
    """

    """
    # root folder of replica 360 output folder
    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/"
    method_name_dict = {0: "DIS", 1: "flownet2", 2: "pwcnet"}
    flow_estimate(root_dir, method_name_dict[0])

    # evaluate and output a *.csv file to the folder of optical flow
    flow_evaluate(root_dir, method_name_dict[0])
