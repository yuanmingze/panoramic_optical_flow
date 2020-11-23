import re
import csv
import os

import pathlib

import sys
sys.path.append("..")

import utility.image_io
import utility.flow_io
import utility.image_evaluate
import utility.flow_evaluate


def optical_flow_test(root_dir):
    """
    test the c
    0) generate diff image
    1) generate the diff dragram
    :param root_dir: the root folder of the optical flow
    """
    warped_image_folder = root_dir + "../warped_images/"
    if not os.path.exists(warped_image_folder):
        os.mkdir(warped_image_folder)

    warped_csv_file_path = warped_image_folder + "error.csv"

    # file name regular expression
    index_digit_re = r"\d{4}"
    rgb_image_fne = re.compile(index_digit_re + r"_rgb.jpg")
    flow_fw_fne = re.compile(index_digit_re + r"_opticalflow_forward.flo")
    flow_bw_fne = re.compile(index_digit_re + r"_opticalflow_backward.flo")

    # scan all file to get the index, and load image and of list
    of_forward_list = {}
    of_backward_list = {}
    image_list = {}
    max_index = -1
    for item in pathlib.Path(root_dir).iterdir():
        index_digt = re.search(index_digit_re, item.name).group()
        if index_digt is None:
            continue

        index_digt = int(index_digt)
        if index_digt > max_index:
            max_index = index_digt

        if not rgb_image_fne.match(item.name) is None:
            image_list[index_digt] = item.name
        elif not flow_fw_fne.match(item.name) is None:
            of_forward_list[index_digt] = item.name
        elif not flow_bw_fne.match(item.name) is None:
            of_backward_list[index_digt] = item.name

    # load 3 image and 4 optical flow, A --> B --> C
    error_csv_file = open(warped_csv_file_path, 'w', newline='')
    error_csv = csv.writer(error_csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    output_csv_header = ["# image_gt_filename", "ssim_A_B_fw", "mse_A_B_fw", "ssim_B_A_bw", "mse_B_A_bw", "ssim_C_B_fw", "mse_C_B_fw", "ssim_B_C_bw", "mse_B_C_bw"]
    error_csv.writerow(output_csv_header)

    frame_number = max_index + 1
    for index in range(0, max_index + 1):
        try:
            rgb_image_A_file_name = root_dir + image_list[(index - 1 + frame_number) % frame_number]
            rgb_image_B_file_name = root_dir + image_list[index]
            rgb_image_C_file_name = root_dir + image_list[(index + 1 + frame_number) % frame_number]
            of_A_B_file_name = root_dir + of_forward_list[(index - 1 + frame_number) % frame_number]
            of_B_C_file_name = root_dir + of_forward_list[index]
            of_B_A_file_name = root_dir + of_backward_list[index]
            of_C_B_file_name = root_dir + of_backward_list[(index + 1 + frame_number) % frame_number]
        except KeyError:
            continue

        if index % 1 == 0:
            print("test image {}".format(rgb_image_B_file_name))

        # load data, warp image & compute the diff
        rgb_image_A_data = utility.image_io.image_read(rgb_image_A_file_name)
        rgb_image_B_data = utility.image_io.image_read(rgb_image_B_file_name)
        rgb_image_C_data = utility.image_io.image_read(rgb_image_C_file_name)

        of_A_B_data = utility.flow_io.readFlowFile(of_A_B_file_name)
        of_B_C_data = utility.flow_io.readFlowFile(of_B_C_file_name)
        of_B_A_data = utility.flow_io.readFlowFile(of_B_A_file_name)
        of_C_B_data = utility.flow_io.readFlowFile(of_C_B_file_name)

        base_name = os.path.splitext(os.path.basename(rgb_image_B_file_name))[0]
        rgb_image_A_B_file_name = warped_image_folder + base_name + "_A_B_fw.jpg"
        rgb_image_B_A_file_name = warped_image_folder + base_name + "_B_A_bw.jpg"
        rgb_image_B_C_file_name = warped_image_folder + base_name + "_B_C_bw.jpg"
        rgb_image_C_B_file_name = warped_image_folder + base_name + "_C_B_fw.jpg"

        # warp image to geerate 4 images, A --> B --> C
        # 1) A->B of + A + forward_warp
        rgb_image_A_B = utility.flow_evaluate.warp_forward(rgb_image_A_data, of_A_B_data)
        utility.image_io.image_save(rgb_image_A_B, rgb_image_A_B_file_name)
        ssim_A_B = utility.image_evaluate.ssim(rgb_image_B_data, rgb_image_A_B)
        mse_A_B = utility.image_evaluate.mse(rgb_image_B_data, rgb_image_A_B)

        # 2) B->A of + A + backward_warp
        rgb_image_B_A = utility.flow_evaluate.warp_backward(rgb_image_A_data, of_B_A_data)
        utility.image_io.image_save(rgb_image_B_A, rgb_image_B_A_file_name)
        ssim_B_A = utility.image_evaluate.ssim(rgb_image_B_data, rgb_image_B_A)
        mse_B_A = utility.image_evaluate.mse(rgb_image_B_data, rgb_image_B_A)

        # 3) C->B of + C + forward_warp
        rgb_image_C_B = utility.flow_evaluate.warp_forward(rgb_image_C_data, of_C_B_data)
        utility.image_io.image_save(rgb_image_C_B, rgb_image_C_B_file_name)
        ssim_C_B = utility.image_evaluate.ssim(rgb_image_B_data, rgb_image_C_B)
        mse_C_B = utility.image_evaluate.mse(rgb_image_B_data, rgb_image_C_B)

        # 4) B->C of + C + backward_warp
        rgb_image_B_C = utility.flow_evaluate.warp_backward(rgb_image_C_data, of_B_C_data)
        utility.image_io.image_save(rgb_image_B_C, rgb_image_B_C_file_name)
        ssim_B_C = utility.image_evaluate.ssim(rgb_image_B_data, rgb_image_B_C)
        mse_B_C = utility.image_evaluate.mse(rgb_image_B_data, rgb_image_B_C)

        # compute the ssim & mse, output to csv file
        row = [image_list[index], ssim_A_B, mse_A_B, ssim_B_A, mse_B_A, ssim_C_B, mse_C_B, ssim_B_C, mse_B_C]
        error_csv.writerow(row)

    error_csv_file.close()


if __name__ == "__main__":
    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/"
    data_dir = root_dir + "replica_seq_data/"
    optical_flow_test(data_dir)
