import pathlib
import subprocess
import copy
import os
from os import listdir
from os.path import isfile, join
import json
import shutil

#
from replica360 import gen_video_path_mp
from replica360 import post_process

from utility import depth_io, image_io

""" the model and texture's root folder"""
# replica_data_root_dir= pathlib.Path("/mnt/sda1/workdata/panoramic_rendering/replica_360/")
# original dataset model and texture
replica_data_root_dir = pathlib.Path("/mnt/sda1/workdata/replica_v1_0/")
replica_mesh_file = "mesh.ply"
replica_texture_file = "textures/"
replica_glass_file = "glass.sur"

""" the root folder of output """
# work_root_dir = pathlib.Path("/mnt/sda1/workdata/panoramic_rendering/replica_360/")
work_root_dir = pathlib.Path("/mnt/sda1/workdata/opticalflow_data/replica_360/")
# work_root_dir = pathlib.Path("/mnt/sda1/workdata/panoramic_rendering/replica_360/")
# work_root_dir = pathlib.Path("/mnt/sda1/workdata/lightfield/GT-Replica-workdata-cubemap/")
config_json_file_name = "config.json"

"""data generating and processing program"""
render_panorama_program_file_path = "/mnt/sda1/workspace_linux/replica360/build/ReplicaSDK/ReplicaOpticalFlow"
render_cubemap_program_file_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/code/cpp/replica360/build/render_cubemap"
# visualize optical flow program
dataconvertion_program_file_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/code/cpp/build/replica_otical_flow_post_process"


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

            
def render_datasets():
    """
    render dataset queue
    """
    # render_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]
    # render_list = ["cube_01_rendering"]
    render_list = ["office_0"]
    # render_list = ["hotel_0","apartment_0", "office_0"]

    # get all folders name
    # for dataset_name in datatset_name_list:
    for dir_item in os.listdir(str(work_root_dir)):
        if not (work_root_dir / dir_item).is_dir() \
                or dir_item not in render_list:
            continue

        # begin generate data
        try:
            print("=======rendering {}==========".format(dir_item))
            # clear folder
            for delete_item in (work_root_dir / dir_item).iterdir():
                if delete_item.name == "config.json":
                    continue
                if delete_item.is_dir():
                    shutil.rmtree(str(delete_item))
                else:
                    os.remove(str(delete_item))
            # load config from json file
            config = None
            config_json_file_path = str(
                work_root_dir / dir_item / config_json_file_name)
            with open(config_json_file_path) as json_file:
                config = json.load(json_file)

            scene_name = config["scene_name"]

            FPS = config["image"]["fps"]
            image_height = config["image"]["height"]
            image_width = config["image"]["width"]
            convert_format_all_enable = config["post_proc"]["convert_format_traj"]
            convert_format_centre_enable = config["post_proc"]["convert_format_center"]
            # the render's viewpoint
            render_view_center = config["render_view"]["center_view"]
            render_view_traj = config["render_view"]["traj_view"]

            msg_file_enable = config["post_proc"]["generate_msg_file"]

            # post_process.image_height = image_height
            # post_process.image_width = image_width
            # post_process.FPS = FPS

            #postfix_res = "_" + str(image_width) + "x" + str(image_height)
            postfix_res = ""
            output_render_seq = "replica_seq_data" + postfix_res
            output_render_center = "replica_center_data" + postfix_res

            path_type = config["camera_traj"]["type"]
            render_type = config["render_type"]

            # genene camera path
            print("---- 1) genene camera path")
            path_csv_file, center_csv_file = gen_video_path_mp.generate_path(str(work_root_dir / dir_item), config)

            # render dataset
            print("---- 2) render dataset")
            if render_type == "panorama":
                print("- render panorama images")
            elif render_type == "cubemap":
                print("- render cubemap images")

            if render_view_center:
                print("- render center viewpoint images")
            elif render_view_traj:
                print("- render trajectory viewpoint images")

            # create the parameters
            render_args_mesh = []
            render_args_mesh.append(str(replica_data_root_dir / scene_name / replica_mesh_file))
            render_args_mesh.append(str(replica_data_root_dir / scene_name / replica_texture_file) + "/")
            render_args_mesh.append(str(replica_data_root_dir / scene_name / replica_glass_file))

            render_args_imageinfo = []
            render_args_imageinfo.append(str(image_width))
            render_args_imageinfo.append(str(image_height))

            # render_seq_process_args = copy.deepcopy(render_args)
            # render camera viewpoint sequence
            if render_view_traj:
                render_args = []
                if render_type == "panorama":
                    render_args.append(render_panorama_program_file_path)
                    render_args = render_args + render_args_mesh
                    render_args.append("y")
                    render_args = render_args + render_args_imageinfo

                    # render output folder
                    pose_output_folder = str(work_root_dir / dir_item / output_render_seq) + "/"
                    render_args.insert(5, pose_output_folder)
                    render_args.insert(4, path_csv_file)

                elif render_type == "cubemap":
                    render_args.append(render_cubemap_program_file_path)
                    render_args = render_args + render_args_mesh
                    render_args.append("n")
                    render_args = render_args + render_args_imageinfo

                    # render output folder
                    pose_output_folder = str(work_root_dir / dir_item / output_render_seq) + "/"
                    render_args.insert(5, pose_output_folder)
                    render_args.insert(4, path_csv_file)

                # run the render porgram
                os.mkdir(pose_output_folder)
                print(render_args)
                render_seq_return = subprocess.run(render_args)

            # render center view and output panoramic images
            if render_view_center:
                #if render_type == "panorama":
                render_args = []
                render_args.append(render_panorama_program_file_path)
                render_args = render_args + render_args_mesh
                render_args.append("y")
                if render_type == "panorama":
                    render_args = render_args + render_args_imageinfo
                elif render_type == "cubemap":
                    render_args = render_args + ["2048", "1024"]
                    print("the center viewpoint render with resolution {}x{}".format(2048, 1024))

                # render output folder
                center_output_folder = str(work_root_dir / dir_item / output_render_center) + "/"
                render_args.insert(5, center_output_folder)
                render_args.insert(4, center_csv_file)
                print(render_args)

                os.mkdir(center_output_folder)
                render_center_return = subprocess.run(render_args)

                # generate the visual depth map
                depth_map_process(center_output_folder)

            # data format convertion
            if convert_format_all_enable or convert_format_centre_enable:
                print("---- 3) convert the file format")

            if convert_format_all_enable:
                print("convert the traj viewpoint file format")
                # convert traj data
                data_convertion_seq_process_args = []
                data_convertion_seq_process_args.append(dataconvertion_program_file_path)
                data_convertion_seq_process_args.append(pose_output_folder)
                data_convertion_seq_process_args.append("n")  # depth_convert_enable
                data_convertion_seq_process_args.append("y")  # optical_flow_convert_enable
                convertion_seq_return = subprocess.run(data_convertion_seq_process_args)

            if convert_format_centre_enable:
                print("convert the center file format")
                data_convertion_center_process_args = []
                data_convertion_center_process_args.append(dataconvertion_program_file_path)
                data_convertion_center_process_args.append(center_output_folder)
                data_convertion_center_process_args.append("y")  # depth_convert_enable
                data_convertion_center_process_args.append("n")  # optical_flow_convert_enable
                convertion_center_return = subprocess.run(data_convertion_center_process_args)

            # create megaparallax style file
            if msg_file_enable:
                print("---- 4) generate megaparallax file")
                post_process.post_process(root_dir=str(work_root_dir / dir_item) + "/", dataset_name=scene_name,
                                          replica_camera_file=path_csv_file, replica_center_camera_file=center_csv_file,
                                          original_traj_dir=pose_output_folder, original_center_dir=center_output_folder,
                                          preprocessing_folder="Preprocessing" + postfix_res + "/", viewer_folder="Viewer" + postfix_res + "/",
                                          input_image_width=image_width, input_image_height=image_height, input_FPS=FPS)

        except Exception as error:
            print(error)


if __name__ == "__main__":
    render_datasets()
