#!/usr/bin/env python

"""
Note it's Python 2. Set up method refere `https://github.com/COATZ/OmniFlowNet`.
Run the script on the folder `OmniFlowNet/models/testing/`.

There are 2 steps: 1) create image list file; 2) get the optical flow.

Input is the `image_list.csv` file.

Each line contain 3 column,

`source_image_relative_path, target_image_relative_path, optical_flow_output_relative_path`


"""


import os, sys, csv
import shutil
import subprocess
from math import ceil

replica_dataset_circ_dirlist = []
replica_dataset_line_dirlist = []
replica_dataset_rand_dirlist = []

opticalflow_mathod = "omniflownet"
scene_list = [
    "apartment_0",
    "apartment_1",
    "frl_apartment_0",
    "frl_apartment_2",
    "frl_apartment_3",
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1"
    ]
for scene_name in scene_list:
    replica_dataset_circ_dirlist.append(scene_name + "_circ_1k_0")
    replica_dataset_line_dirlist.append(scene_name + "_line_1k_0")
    replica_dataset_rand_dirlist.append(scene_name + "_rand_1k_0")

from skimage.transform import resize
import numpy as np
from PIL import Image

def image_file_resize(image_input_filepath, image_output_filepath, resize_ratio = 1.0):
    image_data = np.asarray(Image.open(image_input_filepath))

    image_height = int(image_data.shape[0] * resize_ratio)
    image_width = int(image_data.shape[1] * resize_ratio)

    image_data_resized = resize(image_data, (image_height, image_width),
                                anti_aliasing=False, preserve_range=True).astype(np.uint8)

    # image_io.image_save(image_data_resized, image_output_filepath)
    im = Image.fromarray(image_data_resized)
    im.save(image_output_filepath)


def fs_utility_dir_make(directory):
    # check
    if os.path.exists(directory):
        print("Directory {} exist".format(directory))
    else:
        # create folder
        os.mkdir(directory)


def fs_utility_move_file(src, tar):
    # check
    shutil.move(src, tar)


def create_image_list_omniphoto():
    pass


def create_image_list_replica(replica_dataset_root_dir):
    """Create the replia image list file for OmniFlowNet input script.
    Each image list txt for each scene.
    :param root_dir: the root dir of the rendered replica 
    """


    dataset_dirlist = replica_dataset_circ_dirlist \
                    + replica_dataset_line_dirlist \
                    + replica_dataset_rand_dirlist

    #
    replica_dataset_pano_data_dir = "pano/"

    replica_pano_rgb_image_filename_exp = "{:04d}_rgb_pano.jpg"
    replica_pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    replica_pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"

    replica_dataset_line_start_idx = 4
    replica_dataset_line_end_idx = 8

    replica_dataset_circle_start_idx = 4
    replica_dataset_circle_end_idx = 8

    replica_dataset_rand_start_idx = 4
    replica_dataset_rand_end_idx = 8

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        # each scene
        print("processing the data folder {}".format(pano_image_folder))

        # input index
        if pano_image_folder.find("line") != -1:
            pano_start_idx = replica_dataset_line_start_idx
            pano_end_idx = replica_dataset_line_end_idx
        elif pano_image_folder.find("circ") != -1:
            pano_start_idx = replica_dataset_circle_start_idx
            pano_end_idx = replica_dataset_circle_end_idx
        elif pano_image_folder.find("rand") != -1:
            pano_start_idx = replica_dataset_rand_start_idx
            pano_end_idx = replica_dataset_rand_end_idx
        else:
            print("{} folder naming is wrong".format(pano_image_folder))

        # input image dir
        input_image_root_dir = pano_image_folder + "/" + replica_dataset_pano_data_dir
        fs_utility_dir_make(replica_dataset_root_dir + input_image_root_dir)

        # output flo folder
        output_flo_root_dir_ = pano_image_folder + "/result/" 
        fs_utility_dir_make(replica_dataset_root_dir + output_flo_root_dir_)
        output_flo_root_dir = output_flo_root_dir_ + opticalflow_mathod + "/"
        fs_utility_dir_make(replica_dataset_root_dir + output_flo_root_dir)

        # output image list txt filepath
        image_list_txt_output_dir = replica_dataset_root_dir + opticalflow_mathod + "/"
        fs_utility_dir_make(image_list_txt_output_dir)
        image_list_txt_output_filepath = image_list_txt_output_dir + pano_image_folder + ".txt"
        image_list_txt_file = open(image_list_txt_output_filepath, "w")

        for pano_image_idx in range(pano_start_idx, pano_end_idx):
            for forward_of in [True, False]:
                # 0) load image to CPU memory
                if forward_of:
                    src_erp_image_filepath = replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                    optical_flow_filepath = replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                else:
                    src_erp_image_filepath = replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                    tar_erp_image_filepath = replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                    optical_flow_filepath = replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)

                if pano_image_idx % 3 == 0:
                    print("{} Flow Method: {}\n{}\n{}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))

                # output file path
                src_erp_image_filepath = input_image_root_dir + src_erp_image_filepath
                tar_erp_image_filepath = input_image_root_dir + tar_erp_image_filepath
                optical_flow_filepath = output_flo_root_dir + optical_flow_filepath
                line_txt = "{},{},{}".format(src_erp_image_filepath,tar_erp_image_filepath,optical_flow_filepath)

                image_list_txt_file.write(line_txt + "\n")

        image_list_txt_file.close()


def OmniFlowNet_run(replica_dataset_root_dir):
    """
    Function to estimate the optical flow.
    The original OmniFlowNet test_iter.py script.
    """

    # set the caffe run time envirement
    caffe_bin = 'bin/caffe.bin'
    img_size_bin = 'bin/get_image_size'

    # template = './deploy_MODEL.prototxt' # MODEL = LiteFlowNet2-ft-sintel or LiteFlowNet2-ft-kitti
    # cnn_model = 'MODEL'
    template = './deploy_LiteFlowNet2-ft-sintel_SPHE.prototxt' # MODEL = LiteFlowNet2-ft-sintel or LiteFlowNet2-ft-kitti
    cnn_model = 'LiteFlowNet2-ft-sintel'

    # =========================================================
    def get_image_size(filename):
        # global img_size_bin
        dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([img_size_bin, filename])).split(',')]
        if not len(dim_list) == 2:
            print('Could not determine size of image %s' % filename)
            sys.exit(1)
        return dim_list


    def sizes_equal(size1, size2):
        return size1[0] == size2[0] and size1[1] == size2[1]


    # def check_image_lists(lists):
    #     images = [[], []]

    #     with open(lists[0], 'r') as f:
    #         images[0] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    #     with open(lists[1], 'r') as f:
    #         images[1] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    #     if len(images[0]) != len(images[1]):
    #         print("Unequal amount of images in the given lists (%d vs. %d)" % (len(images[0]), len(images[1])))
    #         sys.exit(1)

    #     if not os.path.isfile(images[0][0]):
    #         print('Image %s not found' % images[0][0])
    #         sys.exit(1)

    #     base_size = get_image_size(images[0][0])

    #     for idx in range(len(images[0])):
    #         print("Checking image pair %d of %d" % (idx+1, len(images[0])))
    #         img1 = images[0][idx]
    #         img2 = images[1][idx]

    #         if not os.path.isfile(img1):
    #             print('Image %s not found' % img1)
    #             sys.exit(1)

    #         if not os.path.isfile(img2):
    #             print('Image %s not found' % img2)
    #             sys.exit(1)

    #         img1_size = get_image_size(img1)
    #         img2_size = get_image_size(img2)

    #         if not (sizes_equal(base_size, img1_size) and sizes_equal(base_size, img2_size)):
    #             print('The images do not all have the same size. (Images: %s or %s vs. %s)\n Please use the pair-mode.' % (img1, img2, images[0][idx]))
    #             sys.exit(1)

    #     return base_size[0], base_size[1], len(images[0])

    my_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(my_dir)

    if not (os.path.isfile(caffe_bin) and os.path.isfile(img_size_bin)):
        print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
        sys.exit(1)

    # 
    # img_files = sys.argv[1:] # list: 1st is 
    # print("Image files: " + str(img_files))


    # Frame-by-frame processing

    # with open(img_files[0], 'r') as f:
    #     images[0] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    # with open(img_files[1], 'r') as f:
    #     images[1] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    output_dir_temporary = "tmp/"

    image_list_txt_root_dir = replica_dataset_root_dir + opticalflow_mathod + "/"
    for image_list_txt_filename in os.listdir(image_list_txt_root_dir):

        # load & parser input and output from image_list file
        if not os.path.isfile(image_list_txt_root_dir + image_list_txt_filename):
            continue

        images = [[], [], []]
        print("parser {}".format(image_list_txt_filename))
        with open(image_list_txt_root_dir + image_list_txt_filename,) as csvfile:
            line_data_list = csv.reader(csvfile, delimiter=',', quotechar='|')
            for line_data in line_data_list:
                if len(line_data) != 3:
                    print('{} include error filename {} '.format(image_list_txt_filename, line_data))
                    exit(0)
                for idx in range(len(line_data)):
                    images[idx].append(line_data[idx])

        # estimate the optical flow
        # for idx in reversed(range(len(images[0]))):
        for idx in range(len(images[0])):
            # check image 
            img1 = replica_dataset_root_dir + images[0][idx]
            img2 = replica_dataset_root_dir + images[1][idx]

            img1_size = get_image_size(img1) #[width, height]
            img2_size = get_image_size(img2)

            if not (sizes_equal(img1_size, img2_size)):
                print('The images do not have the same size. (Images: %s or %s vs. %s)\n Please use the pair-mode.' % (img1, img2, images[0][idx]))
                sys.exit(1)

            if img1_size[1] >= 640:
                img1_new_width = 32 * 32
                img1_new_height = 32 * 16
                print("The image resolution is {}, resize image to {}".format(img1_size, (img1_new_height, img1_new_width)))
                img1_new = img1 + "_{}_{}.jpg".format(img1_new_height, img1_new_width)
                img2_new = img2 + "_{}_{}.jpg".format(img1_new_height, img1_new_width)
                
                image_file_resize(img1, img1_new)
                image_file_resize(img2, img2_new)
                img1 = img1_new
                img2 = img2_new
                img1_size = (img1_new_width,img1_new_height)


            width = img1_size[0]
            height = img1_size[1]

            # Prepare prototxt
            subprocess.call('mkdir -p {}'.format(output_dir_temporary), shell=True)

            with open('{}/img1.txt'.format(output_dir_temporary), "w") as tfile:
                tfile.write("%s\n" % img1)

            with open('{}/img2.txt'.format(output_dir_temporary), "w") as tfile:
                tfile.write("%s\n" % img2)


            divisor = 32.
            adapted_width = ceil(width/divisor) * divisor
            adapted_height = ceil(height/divisor) * divisor
            rescale_coeff_x = width / adapted_width
            rescale_coeff_y = height / adapted_height

            replacement_list = {
                '$ADAPTED_WIDTH': ('%d' % adapted_width),
                '$ADAPTED_HEIGHT': ('%d' % adapted_height),
                '$TARGET_WIDTH': ('%d' % width),
                '$TARGET_HEIGHT': ('%d' % height),
                '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
                '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y),
                '$OUTFOLDER': ('%s' % '"' + output_dir_temporary + '"'),
                '$CNN': ('%s' % '"' + cnn_model + '-"'),
            }

            proto = ''
            with open(template, "r") as tfile:
                proto = tfile.read()

            for r in replacement_list:
                proto = proto.replace(r, replacement_list[r])

            with open('{}/deploy.prototxt'.format(output_dir_temporary), "w") as tfile:
                tfile.write(proto)

            # Run caffe
            args = [caffe_bin, 'test', '-model', '{}/deploy.prototxt'.format(output_dir_temporary),
                    '-weights', '../trained/' + cnn_model + '.caffemodel',
                    '-iterations', str(1),
                    '-gpu', '0']

            cmd = str.join(' ', args)
            print('Executing %s' % cmd)
            subprocess.call(args)

            # rename & move the output *.flo file
            flo_caffe_output_filename = output_dir_temporary + '/' + cnn_model + '-0000000.flo'
            flo_replica_result_filename = replica_dataset_root_dir + images[2][idx]
            print("move file from {} to {}".format(flo_caffe_output_filename, flo_replica_result_filename))
            if idx % 4 == 0:
                print("src image: {}".format(img1))
                print("tar image: {}".format(img2))
                print("flo  file: {}".format(flo_replica_result_filename))

            fs_utility_move_file(flo_caffe_output_filename, flo_replica_result_filename)
            
            # if idx > 0:
            #     os.rename(img_files[2] + '/' + cnn_model + '-0000000.flo', img_files[2] + '/' + cnn_model +'-' + '{0:07d}'.format(idx) + '.flo')

        # print('\nThe resulting FLOW is stored in CNN-NNNNNNN.flo')


if __name__ == "__main__":
    data_root_dir = "/home/mingze/sda1/workdata/opticalflow_data_bmvc_2021/"
    image_list_txt_root_dir = data_root_dir + opticalflow_mathod + "/"
    task_list = [2]
    if 0 in task_list:
        # create replica *.txt image list file
        create_image_list_replica(data_root_dir)
    if 1 in task_list:
        # create omniphoto *.txt image list file
        create_image_list_omniphoto(data_root_dir)
    if 2 in task_list:
        # estimate the panoramic optical flow.
        OmniFlowNet_run(data_root_dir)
