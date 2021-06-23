import os
import sys

dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dir_root = os.path.dirname(dir_scripts) # parent dir
print(dir_scripts)
sys.path.append(dir_scripts)
sys.path.append(os.path.join(dir_scripts, "test"))
sys.path.append(os.path.join(dir_scripts, "utility"))

from utility import replica_util
from utility import fs_utility

import configuration 
from main import ReplicaPanoDataset

"""
The code to generate the img1.txt, img2.txt and out.txt for PWC-Net (caffe).
"""

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


def generate_input_txt_bmvc_omniphoto(root_dir, txt_output_dir, flo_output_dir):

    datascene_list= 

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


def generate_input_txt_bmvc_replica(replica_dataset, txt_output_dir):
    """Get the our and DIS's result in replica. """
    img1_txt_path = txt_output_dir + "img1.txt"
    img2_txt_path = txt_output_dir + "img2.txt"
    out_txt_path = txt_output_dir + "out.txt"

    img1_txt_file =  open(img1_txt_path, "w") 
    img2_txt_file = open(img2_txt_path, "w") 
    out_txt_file = open(out_txt_path, "w")

    opticalflow_mathod = "pwcnet"

    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        print("processing the data folder {}".format(pano_image_folder))
        # input dir
        input_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir + "/"
        # input index
        if pano_image_folder.find("line") != -1:
            pano_start_idx = replica_dataset.line_start_idx
            pano_end_idx = replica_dataset.line_end_idx
        elif pano_image_folder.find("circ") != -1:
            pano_start_idx = replica_dataset.circle_start_idx
            pano_end_idx = replica_dataset.circle_end_idx
        else:
            print("{} folder naming is wrong".format(pano_image_folder))

        # output folder
        output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
        output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            for forward_of in [True, False]:
                # 0) load image to CPU memory
                if forward_of:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)

                else:
                    src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                    optical_flow_filepath = replica_dataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)


                if pano_image_idx % 2 == 0:
                    print("{} Flow Method: {}\n{}\n{}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))


            # output file path
            img1_txt_file.write(input_filepath + src_erp_image_filepath + "\n")
            img2_txt_file.write(input_filepath + tar_erp_image_filepath + "\n")
            out_txt_file.write( output_dir + optical_flow_filepath + "\n")


    img1_txt_file.close()
    img2_txt_file.close()
    out_txt_file.close()



if __name__ == "__main__":
    # dataset_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]

    # for dataset_name in dataset_list:
    #     print("-- {} input files.".format(dataset_name))
    #     # the rgb images folder
    #     root_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/" + dataset_name + "/replica_seq_data/"

    #     # the PWC output flo files folder
    #     # the output img1.txt img2.txt and out.txt files folder
    #     txt_output_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360/" + dataset_name + "/pwcnet/"
    #     flo_output_dir = txt_output_dir
    #     generate_input_txt(root_dir, txt_output_dir, flo_output_dir)
    #     visual_of(flo_output_dir)

    txt_output_dir = "/mnt/sda1/"
    generate_input_txt_bmvc_replica(ReplicaPanoDataset, txt_output_dir)
