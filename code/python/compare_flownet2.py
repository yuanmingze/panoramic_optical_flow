import io
import pathlib
import re
import os

from utility import replica_util
from utility import flow_io
from utility import flow_vis
from utility import image_io


def generate_listfile(root_dir, of_txt_path, flo_output_dir):
    """
    create the listfile with line "x.png y.png z.flo"
    :param root_dir: the root dir of the rendered replica 
    """
    if not os.path.exists(flo_output_dir) or not os.path.isdir(flo_output_dir):
        os.mkdir(flo_output_dir)

    # of_txt_path = txt_output_dir + "replica_listfile.txt"

    min_index, max_index, image_list, of_forward_list, of_backward_list = replica_util.scene_of_folder(root_dir)

    with open(of_txt_path, "w") as of_txt_file:

        for index in range(min_index, max_index + 1):
            if index % 20 == 0:
                print("{} : {}".format(index, image_list[index]))

            index_next = index + 1
            if index_next > max_index:
                index_next = min_index + max_index - index
            index_previous = index - 1
            if index_previous < min_index:
                index_previous = max_index + min_index - index

            # forward optical flow
            of_txt_file.write(root_dir + image_list[index] + " "
                              + root_dir + image_list[index_next] + " "
                              + flo_output_dir + of_forward_list[index] + "\n")

            # backward optical flow
            of_txt_file.write(root_dir + image_list[index] + " "
                              + root_dir + image_list[index_previous] + " "
                              + flo_output_dir + of_backward_list[index] + "\n")

    print("generate replica listfile.txt")


if __name__ == "__main__":
    dataset_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]

    for dataset_name in dataset_list:
        # the rgb images folder
        root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/"+dataset_name+"/replica_seq_data/"
        flo_output_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/"+dataset_name+"/flownet2/"
        txt_output_dir = flo_output_dir + "replica_listfile.txt"
        generate_listfile(root_dir, txt_output_dir, flo_output_dir)

