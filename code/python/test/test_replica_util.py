import re
import csv
import os
import re

import pathlib

import configuration as config
import flow_warp
from utility import projection_cubemap as proj_cm
from utility import image_io
from utility import depth_io
from utility import flow_io
from utility import flow_vis
from utility import image_evaluate
from utility import flow_evaluate
from utility import replica_util


from replica360 import configuration as replica360config

def test_pano_optical_flow():
    """ """
    # read rgb and optical flow
    data_root = config.TEST_data_root_dir + "replica_360_cubemap/office_0_line_pano/"
    # data_root = config.TEST_data_root_dir + "replica_360_cubemap/office_0_line_pano/"
    rgb_filepath = data_root + "0001_rgb_pano.jpg"
    rgb_data = image_io.image_read(rgb_filepath)

    flo_filepath = data_root + "0001_motionvector_backward.flo"
    flo_data = flow_io.read_flow_flo(flo_filepath)

    # image_io.image_show(flow_vis.flow_to_color(flo_data))

    # warp data
    rgb_data_warp = flow_warp.warp_forward(rgb_data, flo_data, True)
    rgb_data_filepath = data_root + "0001_rgb_pano_warp.jpg"
    print("Output to {}".format(rgb_data_filepath))
    image_io.image_save(rgb_data_warp, rgb_data_filepath)
    # image_io.image_show(rgb_data_warp)


def test_cubemap_optical_flow_withwraparound():
    """Load optical flow and detect the wrap around pixel from dpt file. """

    # # test optical flow warp around
    dataroot_dir = config.TEST_data_root_dir
    optical_flow_dir = dataroot_dir + "replica_360_cubemap/office_0_line_cubemap/"

    # 1) load the flow file to memory
    face_flows = []
    rgb_data_list = []
    # face_flow_depths = []
    face_flows_wraparound = []
    cubemap_face_abbre = ["R", "L", "U", "D", "F", "B"]
    face_rgb_name_expression = "0000_{}_rgb.jpg"
    face_gt_flow_padding_name_expression = "0001_{}_motionvector_backward.flo"
    face_gt_flow_depth_padding_name_expression = "0001_{}_motionvector_backward.flo.dpt"
    for index in cubemap_face_abbre:
        # read optical flow *.flo files
        cubemap_flow_path = optical_flow_dir + face_gt_flow_padding_name_expression.format(index)
        face_flow_data = flow_io.read_flow_flo(cubemap_flow_path)
        face_flows.append(face_flow_data)
        # read optical flow target points depth value *.dpt file
        cubemap_flow_depth_path = optical_flow_dir + face_gt_flow_depth_padding_name_expression.format(index)
        # face_flow_depths.append(depth_io.read_dpt(cubemap_flow_depth_path))
        of_depth_data = depth_io.read_dpt(cubemap_flow_depth_path)
        # image_io.image_show(of_depth_data)
        wraparound_data = replica_util.opticalflow_warparound(of_depth_data)
        face_flows_wraparound.append(wraparound_data)
        
        if index == 'L':
            image_io.image_show(flow_vis.flow_to_color(face_flow_data,min_ratio=0.2, max_ratio=0.8, add_bar = True))
            flow_vis.flow_value_to_color(face_flow_data)
            image_io.image_show(of_depth_data)
            # rgb_data = image_io.image_read(optical_flow_dir + face_rgb_name_expression.format(index))
            # rgb_data_warp = flow_warp.warp_forward_padding(rgb_data, face_flow_data, padding_x=1000, padding_y=1000)
            # import ipdb; ipdb.set_trace()
            # image_io.image_show(rgb_data_warp)

    # 2) test stitch the cubemap flow. Note enable test 3
    padding_size = 0.0
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows, face_flows_wraparound, erp_image_height=480, padding_size=padding_size, wrap_around=True)

    import flow_postproc
    erp_flow_stitch = flow_postproc.erp_of_wraparound(erp_flow_stitch)
    # erp_flow_stitch = flow_postproc.erp_of_unwraparound(erp_flow_stitch)

    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    flow_vis.flow_value_to_color(erp_flow_stitch)
    erp_flow_stitch_name = optical_flow_dir + "0001_motionvector_backward.flo.jpg"
    image_io.image_save(face_flow_vis, erp_flow_stitch_name)
    # forward warp the source image
    rgb_data = image_io.image_read(optical_flow_dir + "0001_rgb_pano.jpg")
    from utility import image_utility
    rgb_data = image_utility.image_resize(rgb_data, erp_flow_stitch.shape[:2])
    rgb_data_forward_warp = flow_warp.warp_forward(rgb_data, erp_flow_stitch, wrap_around=True)
    image_io.image_show(rgb_data_forward_warp)
    image_io.image_save(rgb_data_forward_warp, optical_flow_dir + "0001_rgb_pano_warp" + ".jpg")


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
        rgb_image_A_data = image_io.image_read(rgb_image_A_file_name)
        rgb_image_B_data = image_io.image_read(rgb_image_B_file_name)
        rgb_image_C_data = image_io.image_read(rgb_image_C_file_name)

        of_A_B_data = flow_io.readFlowFile(of_A_B_file_name)
        of_B_C_data = flow_io.readFlowFile(of_B_C_file_name)
        of_B_A_data = flow_io.readFlowFile(of_B_A_file_name)
        of_C_B_data = flow_io.readFlowFile(of_C_B_file_name)

        base_name = os.path.splitext(os.path.basename(rgb_image_B_file_name))[0]
        rgb_image_A_B_file_name = warped_image_folder + base_name + "_A_B_fw.jpg"
        rgb_image_B_A_file_name = warped_image_folder + base_name + "_B_A_bw.jpg"
        rgb_image_B_C_file_name = warped_image_folder + base_name + "_B_C_bw.jpg"
        rgb_image_C_B_file_name = warped_image_folder + base_name + "_C_B_fw.jpg"

        # warp image to geerate 4 images, A --> B --> C
        # 1) A->B of + A + forward_warp
        rgb_image_A_B = flow_evaluate.warp_forward(rgb_image_A_data, of_A_B_data)
        image_io.image_save(rgb_image_A_B, rgb_image_A_B_file_name)
        ssim_A_B = image_evaluate.ssim(rgb_image_B_data, rgb_image_A_B)
        mse_A_B = image_evaluate.mse(rgb_image_B_data, rgb_image_A_B)

        # 2) B->A of + A + backward_warp
        rgb_image_B_A = flow_evaluate.warp_backward(rgb_image_A_data, of_B_A_data)
        image_io.image_save(rgb_image_B_A, rgb_image_B_A_file_name)
        ssim_B_A = image_evaluate.ssim(rgb_image_B_data, rgb_image_B_A)
        mse_B_A = image_evaluate.mse(rgb_image_B_data, rgb_image_B_A)

        # 3) C->B of + C + forward_warp
        rgb_image_C_B = flow_evaluate.warp_forward(rgb_image_C_data, of_C_B_data)
        image_io.image_save(rgb_image_C_B, rgb_image_C_B_file_name)
        ssim_C_B = image_evaluate.ssim(rgb_image_B_data, rgb_image_C_B)
        mse_C_B = image_evaluate.mse(rgb_image_B_data, rgb_image_C_B)

        # 4) B->C of + C + backward_warp
        rgb_image_B_C = flow_evaluate.warp_backward(rgb_image_C_data, of_B_C_data)
        image_io.image_save(rgb_image_B_C, rgb_image_B_C_file_name)
        ssim_B_C = image_evaluate.ssim(rgb_image_B_data, rgb_image_B_C)
        mse_B_C = image_evaluate.mse(rgb_image_B_data, rgb_image_B_C)

        # compute the ssim & mse, output to csv file
        row = [image_list[index], ssim_A_B, mse_A_B, ssim_B_A, mse_B_A, ssim_C_B, mse_C_B, ssim_B_C, mse_B_C]
        error_csv.writerow(row)

    error_csv_file.close()


if __name__ == "__main__":
    root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/"
    data_dir = root_dir + "replica_seq_data/"

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=int, help='the task index')

    args = parser.parse_args()
    
    task_list = []
    task_list.append(args.task)

    if 0 in task_list:
        optical_flow_test(data_dir)
    if 1 in task_list:
        test_cubemap_optical_flow_withwraparound()
    if 2 in task_list:
        test_pano_optical_flow()
