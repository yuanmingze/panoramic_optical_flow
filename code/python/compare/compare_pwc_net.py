import io
import pathlib
import re
import os

from utility import replica_util
from utility import flow_io
from utility import flow_vis
from utility import image_io


def generate_input_txt(root_dir, txt_output_dir, flo_output_dir):
    """
    create the img1.txt img2.txt and out.txt files for pwc_net proc_images.py script.
    :param root_dir: the root dir of the rendered replica 
    """
    if not os.path.exists(flo_output_dir) or not os.path.isdir(flo_output_dir):
        os.mkdir(flo_output_dir)

    img1_txt_path = txt_output_dir + "img1.txt"
    img2_txt_path = txt_output_dir + "img2.txt"
    out_txt_path = txt_output_dir + "out.txt"

    min_index, max_index, image_list, of_forward_list, of_backward_list = replica_util.scene_of_folder(root_dir)

    with open(img1_txt_path, "w") as img1_txt_file, \
            open(img2_txt_path, "w") as img2_txt_file,\
            open(out_txt_path, "w") as out_txt_file:

        for index in range(min_index, max_index + 1):
            index_next = index + 1
            if index_next > max_index:
                index_next = min_index + max_index - index
            index_previous = index - 1
            if index_previous < min_index:
                index_previous = max_index + min_index - index

            # forward optical flow
            img1_txt_file.write(root_dir + "/" + image_list[index] + "\n")
            img2_txt_file.write(root_dir + "/" + image_list[index_next] + "\n")
            out_txt_file.write(flo_output_dir + "/" + of_forward_list[index] + "\n")

            # backward optical flow
            img1_txt_file.write(root_dir + "/" + image_list[index] + "\n")
            img2_txt_file.write(root_dir + "/" + image_list[index_previous] + "\n")
            out_txt_file.write(flo_output_dir + "/" + of_backward_list[index] + "\n")

    print("generate img1.txt, img2.txt and out.txt files")


def visual_of(root_dir):
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
    dataset_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]

    for dataset_name in dataset_list:
        print("-- {} input files.".format(dataset_name))
        # the rgb images folder
        root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/" + dataset_name + "/replica_seq_data/"

        # the PWC output flo files folder
        # the output img1.txt img2.txt and out.txt files folder
        txt_output_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/" + dataset_name + "/pwcnet/"
        flo_output_dir = txt_output_dir
        generate_input_txt(root_dir, txt_output_dir, flo_output_dir)
        visual_of(flo_output_dir)
