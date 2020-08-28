import re
import os
import sys
import pathlib

from utility import flow_estimate, flow_io

"""
Evaluate the quality of optical flow in panoramic images optical flow.
"""


def flow_estimate(root_dir):
    """
    estimate optical flow with specified method.
    """
    output_dir = root_dir + "../DIS_flow/"
    if os.path.exists(output_dir):
        os.mkdir(output_dir)

    # search all rgb image
    index_digit_re = r"\d{4}"
    index_digit_format = r"{:04d}"
    rgb_image_fne = re.compile(index_digit_re + r"_rgb.jpg")
    flow_fw_fne = re.compile(index_digit_re + r"_opticalflow_forward.flo")
    flow_bw_fne = re.compile(index_digit_re + r"_opticalflow_backward.flo")

    image_list = []
    min_index = sys.maxsize
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
            image_list.append(item.name)

    image_list.sort()
    frame_number = len(image_list)

    # compute the optical flow A --> B and B --> A
    for index in range(min_index, max_index + 1):
        rgb_image_A_file_name = root_dir + image_list[(index - 1 + index) % frame_number]
        rgb_image_B_file_name = root_dir + image_list[index]
        rgb_image_A_data = utility.image_io.image_read(rgb_image_A_file_name)
        rgb_image_B_data = utility.image_io.image_read(rgb_image_B_file_name)

        if index % 2 == 0:
            print("compute optical flow {} to {}".format(rgb_image_A_file_name, rgb_image_B_file_name))
            print("compute optical flow {} to {}".format(rgb_image_B_file_name, rgb_image_A_file_name))

        flow_A_B_data = flow_estimate.DIS(rgb_image_A_data, rgb_image_B_data)
        flow_A_B_file_name = flow_fw_fne.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_A_B_data, output_dir + flow_A_B_file_name)

        flow_B_A_data = flow_estimate.DIS(rgb_image_B_data, rgb_image_A_data)
        flow_B_A_file_name = flow_bw_fne.replace(index_digit_re, index_digit_format.format(index))
        flow_io.writeFlowFile(flow_B_A_data, output_dir + flow_B_A_file_name)


def flow_evaluate(root_dir):
    """
    evaluate the quality of estimated optical flow.
    """
    flow_gt_dir = root_dir + "/"
    flow_estimated_dir = root_dir + "/"

    # estimate and output to csv file



if __name__ == "__main__":
    """

    """
    pass
