import re
import os
import sys
import csv
import pathlib

import sys
sys.path.append("..")

from utility import flow_io
from utility import image_io
from utility import flow_vis
from utility import replica_util
from utility import flow_estimate


"""
Evaluate the quality of optical flow in panoramic images optical flow.
"""

INDEX_MAX = sys.maxsize

def flow_estimate_dis(input_dir, flo_output_dir):
    """
    estimate optical flow with specified method.
    """
    if not os.path.exists(flo_output_dir):
        os.mkdir(flo_output_dir)

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
        flow_A_B_data = flow_estimate.of_methdod_DIS(rgb_image_A_data, rgb_image_B_data)
        flow_A_B_file_name = flow_fw_fne_str.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_A_B_data, flo_output_dir + flow_A_B_file_name)

        flow_visual_data = flow_vis.flow_to_color(flow_A_B_data)
        image_io.image_save(flow_visual_data, flo_output_dir + os.path.splitext(flow_A_B_file_name)[0] + "_vis.jpg")

        flow_B_A_data = flow_estimate.of_methdod_DIS(rgb_image_B_data, rgb_image_A_data)
        flow_B_A_file_name = flow_bw_fne_str.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_B_A_data, flo_output_dir + flow_B_A_file_name)

        flow_visual_data = flow_vis.flow_to_color(flow_B_A_data)
        image_io.image_save(flow_visual_data, flo_output_dir + os.path.splitext(flow_B_A_file_name)[0] + "_vis.jpg")


def visual_of(root_dir):
    """
    visual all flo files.
    """
    flow_file_str = r"[0-9]*_opticalflow_[a-zA-Z]*.`flo"
    flow_file_re = re.compile(flow_file_str)
    of_list = []

    # load all optical flow file
    for item in pathlib.Path(root_dir).iterdir():
        if not flow_file_re.match(item.name) is None:
            of_list.append(item.name)

    of_min = -300
    of_max = 300

    # visualization optical flow
    for index in range(len(of_list)):
        item = of_list[index]
        of_data = flow_io.readFlowFile(root_dir + item)
        of_data_vis = flow_vis.flow_to_color(of_data, [of_min, of_max])
        of_vis_file_name = item.replace(".flo", ".jpg")
        image_io.image_save(of_data_vis, root_dir + of_vis_file_name)

        if index % 10 == 0:
            print("processing {}: {} to {}".format(index, item, of_vis_file_name))


if __name__ == "__main__":

    # dataset_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]
    dataset_list = ["apartment_0"]

    for dataset_name in dataset_list:
        # root folder of replica 360 output folder
        root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/"+dataset_name+"/replica_seq_data/"
        flo_output_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/"+dataset_name+"/dis/"
        flow_estimate_dis(root_dir, flo_output_dir)
        visual_of(flo_output_dir)