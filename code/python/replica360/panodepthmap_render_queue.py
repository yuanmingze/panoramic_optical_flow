import configuration
from configuration import ReplicaConfig

import os
from os import listdir
from os.path import isfile, join
import sys
import argparse
import json
import pathlib
import subprocess
import platform

from utility import fs_utility
from utility import depth_io, image_io

from replica360 import create_rendering_pose as gen_video_path_mp

dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(dir_scripts)
sys.path.append(dir_scripts)

from utility.logger import Logger
log = Logger(__name__)
log.logger.propagate = False

"""
Generate the 2K image sequence for the 360 depth map experiments.
"""

def depth_map_process(dir_path):
    """
    read the *.dpt file of depth map, and convert to *.png, generate the visualized depth image
    """
    for file_name in listdir(dir_path):
        if not isfile(join(dir_path, file_name)):
            continue

        filename, file_extension = os.path.splitext(file_name)
        if file_extension == ".dpt":
            depth_data = depth_io.read_dpt(join(dir_path, file_name))
            depth_io.write_png(depth_data, join(dir_path, filename + ".png"))

            # read and visual and convert
            depth_data_vis = depth_io.depth_visual(depth_data)
            image_io.image_save(depth_data_vis, join(dir_path, filename + "_vis.jpg"))


def vis_folder(data_root_dir, sub_folder_name = "replica_seq_data"):
    """ Visualize the dpt files

    :param data_root_dir: the root folder of the rendering output folder.
    :type data_root_dir: str
    """    
    # data format convertion
    for dir_item in os.listdir(data_root_dir):
        if not os.path.isdir(os.path.join(data_root_dir, dir_item)):
            continue

        log.info("visualized *.dpt files in fodler {}".format(dir_item))

        scene_output_root_dir = data_root_dir + dir_item  + "/" + sub_folder_name + "/"
        dpt_filelist = fs_utility.list_files(scene_output_root_dir,".dpt")

        # visualized file
        for file_item in dpt_filelist:
            depth_data = depth_io.read_dpt(scene_output_root_dir + file_item)
            output_path =  scene_output_root_dir + file_item + ".jpg"
            if os.path.exists(output_path):
                continue
            depth_io.depth_visual_save(depth_data, output_path)



def render_replica_config_copy(data_root_dir_pre, data_root_dir_new, replace_words_pairs = None):
    """ Make folder and copy the configuration and camera path file.

    :param data_root_dir_pre: the root folder of the previous output folder.
    :type data_root_dir_pre: str
    :param data_root_dir_new: the target folder path.
    :type data_root_dir_new: str
    """
    for dir_item in os.listdir(data_root_dir_pre):
        if not os.path.isdir(os.path.join(data_root_dir_pre, dir_item)):
            continue

        scene_output_root_dir_pre = data_root_dir_pre + dir_item  + "/"
        json_config_filelist = fs_utility.list_files(scene_output_root_dir_pre,".json")
        csv_config_filelist = fs_utility.list_files(scene_output_root_dir_pre, ".csv")

        if not json_config_filelist and not csv_config_filelist:
            continue

        # 1) make folder and copy file
        log.info("make folder and copy files from {}".format(scene_output_root_dir_pre))

        # 1-1) make folder
        scene_output_root_dir_new = data_root_dir_new + dir_item + "/"
        fs_utility.dir_make(scene_output_root_dir_new)

        # 1-2) copy file
        for file_item in json_config_filelist + csv_config_filelist:
            src_filepath = data_root_dir_pre + dir_item + "/" + file_item
            tar_filepath = data_root_dir_new + dir_item + "/" + file_item
            log.info(f"copy file from {src_filepath} to {tar_filepath}")
            # shutil.copy(src_filepath, tar_filepath)
            fs_utility.copy_replace(src_filepath, tar_filepath, replace_words_pairs)


def render_replica_datasets():
    """
    render dataset queue
    """
    replica_dataset_config = ReplicaConfig()
    replica_dataset_config.renderMotionVectorEnable = False
    replica_dataset_config.renderDepthEnable = True
    replica_dataset_config.renderRGBEnable = True

    """ the model and texture's root folder"""
    # original dataset model and texture
    replica_data_root_dir = pathlib.Path(replica_dataset_config.replica_data_root_dir)

    """ the root folder of output """
    if platform.system() == "Windows":
        output_root_dir = "D:/workdata/panoramic_rendering/replica_360/"
    elif platform.system() == "Linux":
        output_root_dir = "/mnt/sda1/workdata/InstaOmniDepth/replica360_2k/"
    output_root_dir = pathlib.Path(output_root_dir)

    config_json_file_name = "config.json"

    """data generating and processing program"""
    render_panorama_program_file_path = replica_dataset_config.render_panorama_program_filepath

    # # render_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]
    # render_list = replica_dataset_config.replica_scene_name_list

    # get all folders name
    # for dataset_name in datatset_name_list:
    for dir_item in os.listdir(str(output_root_dir)):
        if not (output_root_dir / dir_item).is_dir():
            continue

        # begin generate data
        try:
            print("=======rendering {}==========".format(dir_item))
            # 0) load config from json file
            replica_scene_config = None
            config_json_file_path = str(output_root_dir / dir_item / config_json_file_name)
            with open(config_json_file_path) as json_file:
                replica_scene_config = json.load(json_file)

            scene_name = replica_scene_config["scene_name"]
            if not scene_name in replica_dataset_config.replica_scene_name_list:
                log.warning("The dat")
                continue
            # FPS = config["image"]["fps"]
            image_height = replica_scene_config["image"]["height"]
            image_width = replica_scene_config["image"]["width"]

            if image_width != image_height * 2:
                log.error(f"The image {image_width} is not the twice of {image_height}!")

            # the render's viewpoint
            render_view_center = replica_scene_config["render_view"]["center_view"]
            if render_view_center:
                log.warning("The pano-depthmap project do not need center view data.")
            render_view_traj = replica_scene_config["render_view"]["traj_view"]
            if not render_view_traj:
                log.error("Will not rendering data with camera path.")

            #postfix_res = "_" + str(image_width) + "x" + str(image_height)
            postfix_res = ""
            output_render_seq_folder = "replica_seq_data" + postfix_res

            render_type = replica_scene_config["render_type"]
            if render_type != "panorama":
                msg = "The render type {} is not panorama!".format(render_type)
                log.error(msg)

            # 1) genene camera path
            print("---- 1) genene camera path")
            csv_file_list = fs_utility.list_files(str(output_root_dir / dir_item), ".csv")
            if not csv_file_list:
                log.info("Do not have exist camera path csv file in folder {}".format(str(output_root_dir / dir_item)))
                path_csv_file, _ = gen_video_path_mp.generate_path(str(output_root_dir / dir_item), replica_scene_config)
            else:
                csv_file_list.sort()
                path_csv_file = csv_file_list[0]
                path_csv_file = str(output_root_dir / dir_item) + "/" + path_csv_file
                log.info("use exited camera path csv file {} rendering.".format(path_csv_file))

            # 2) render dataset
            print("---- 2) render dataset")

            # 2-0) create the replica rendering CLI parameters
            render_args_mesh = []
            render_args_mesh.append("--data_root")
            render_args_mesh.append(str(replica_data_root_dir / scene_name) + "/")
            render_args_mesh.append("--meshFile")
            render_args_mesh.append(replica_dataset_config.replica_mesh_file)
            render_args_mesh.append("--atlasFolder")
            render_args_mesh.append(replica_dataset_config.replica_texture_file + "/")
            render_args_mesh.append("--mirrorFile")
            render_args_mesh.append(replica_dataset_config.replica_glass_file)

            render_args_imageinfo = []
            render_args_imageinfo.append("--imageHeight")
            render_args_imageinfo.append(str(image_height))

            render_args_texture_params = []
            render_args_texture_params.append("--texture_exposure")
            render_args_texture_params.append(str(replica_scene_config["render_params"]["texture_exposure"]))
            render_args_texture_params.append("--texture_gamma")
            render_args_texture_params.append(str(replica_scene_config["render_params"]["texture_gamma"]))
            render_args_texture_params.append("--texture_saturation")
            render_args_texture_params.append(str(replica_scene_config["render_params"]["texture_saturation"]))

            render_args_render_data = []
            render_args_render_data.append("--renderRGBEnable=" + str(replica_dataset_config.renderRGBEnable))
            render_args_render_data.append("--renderDepthEnable=" + str(replica_dataset_config.renderDepthEnable))
            render_args_render_data.append("--renderMotionVectorEnable=" + str(replica_dataset_config.renderMotionVectorEnable))

            # 2-1) render camera viewpoint sequence
            # render_seq_process_args = copy.deepcopy(render_args)
            render_args = []
            render_args.append(render_panorama_program_file_path)
            render_args = render_args + render_args_mesh +render_args_imageinfo + render_args_texture_params + render_args_render_data
            # add camera path file
            render_args.append("--cameraPoseFile")
            render_args.append(path_csv_file)
            # render output folder
            result_output_folder = str(output_root_dir / dir_item / output_render_seq_folder) + "/"
            render_args.append("--outputDir")
            render_args.append(result_output_folder)
            fs_utility.dir_make(result_output_folder)
            print(render_args)
            render_seq_return = subprocess.run(render_args)

        except Exception as error:
            print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', metavar='N', type=int, nargs='+', help='the task list')

    args = parser.parse_args()
    print (sys.argv[1:])

    test_list = args.task[:]

    if 1 in test_list:
        replace_words_pairs = {}
        replace_words_pairs["cubemap"] = "panorama"
        replace_words_pairs["\"width\": 320"] = "\"width\": 2048"
        replace_words_pairs["\"height\": 320"] = "\"height\": 1024"

        render_config_dir_pre = "/home/mingze/sda1/workdata/opticalflow_data_bmvc_2021/"
        render_config_dir_new = "/home/mingze/sda1/workdata/InstaOmniDepth/replica360_2k/"

        fs_utility.dir_rm(render_config_dir_new)
        fs_utility.dir_make(render_config_dir_new)
        render_replica_config_copy(render_config_dir_pre, render_config_dir_new, replace_words_pairs)
    if 2 in test_list:
        render_replica_datasets()

    if 3 in test_list:
        data_root_dir = "/home/mingze/sda1/workdata/InstaOmniDepth/replica360_2k/"
        vis_folder(data_root_dir, sub_folder_name = "replica_seq_data")