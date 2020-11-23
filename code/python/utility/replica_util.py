import re
import sys
import pathlib

INDEX_MAX = sys.maxsize

def scene_of_folder(root_dir):
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
