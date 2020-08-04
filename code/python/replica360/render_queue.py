import pathlib
import subprocess
import copy
import os
import json
import shutil

# 
import gen_video_path_mp
import post_process

""" the model and texture's root folder"""
# replica_data_root_dir= pathlib.Path("/mnt/sda1/workdata/panoramic_rendering/replica_360/")
replica_data_root_dir= pathlib.Path("/mnt/sda1/workdata/replica_v1_0/") # original dataset model and texture 
replica_mesh_file = "mesh.ply"
replica_texture_file = "textures/"
replica_glass_file = "glass.sur"

""" the root folder of output """
# work_root_dir = pathlib.Path("/mnt/sda1/workdata/panoramic_rendering/replica_360/")
work_root_dir = pathlib.Path("/mnt/sda1/workdata/opticalflow_data/replica_360/")

config_json_file_name = "config.json"

"""data generating and processing program"""
render_panorama_program_file_path = "/mnt/sda1/workspace_linux/replica360/build/ReplicaSDK/ReplicaOpticalFlow"
render_cubemap_program_file_path = ""
# visualize optical flow
dataconvertion_program_file_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/code/cpp/build/replica_otical_flow_post_process"

def render_datasets():
    """
    render dataset queue
    """

    #render_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]
    render_list = ["hotel_0","apartment_0", "office_0"]

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
            config_json_file_path = str(work_root_dir / dir_item / config_json_file_name)
            with open(config_json_file_path) as json_file:
                config = json.load(json_file)

            scene_name = config["scene_name"]

            FPS = config["image"]["fps"]
            image_height= config["image"]["height"]
            image_width = config["image"]["width"]
            convert_format_all_enable = config["post_proc"]["convert_format_all"]
            convert_format_centre_enable = config["post_proc"]["convert_format_center"]

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
            path_csv_file, center_csv_file = \
                gen_video_path_mp.generate_path(str(work_root_dir / dir_item), config)

            # render dataset
            print("---- 2) render dataset")
            render_args = []
            if render_type == "panorama":
                render_args.append(render_panorama_program_file_path)
            elif render_type == "cubemap":
                render_args.append(render_cubemap_program_file_path)
            render_args.append(str(replica_data_root_dir / scene_name / replica_mesh_file))
            render_args.append(str(replica_data_root_dir / scene_name / replica_texture_file) + "/")
            render_args.append(str(replica_data_root_dir / scene_name / replica_glass_file))
            if render_type == "panorama":#path_type == "circle":
                render_args.append("y") 
            elif render_type == "cubemap":#path_type == "grid":
                render_args.append("n") 
            render_args.append(str(image_width))
            render_args.append(str(image_height))

            # render camera view sequence
            render_seq_process_args = copy.deepcopy(render_args)
            pose_output_folder = str(work_root_dir / dir_item / output_render_seq) + "/"
            render_seq_process_args.insert(5, pose_output_folder) # render output folder
            render_seq_process_args.insert(4, path_csv_file)
            os.mkdir(pose_output_folder)
            print(render_seq_process_args)
            render_seq_return = subprocess.run(render_seq_process_args)

            # render center view
            center_output_folder = str(work_root_dir / dir_item / output_render_center) + "/"
            if render_type == "panorama":
                render_center_process_args = copy.deepcopy(render_args)
                render_center_process_args.insert(5, center_output_folder)  # render output folder
                render_center_process_args.insert(4, center_csv_file)
                os.mkdir(center_output_folder)
                render_center_return = subprocess.run(render_center_process_args)

            # data format convertion
            print("---- 3) convert the file format")
            if convert_format_all_enable:
                print("convert the all file format")
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
            print("---- 4) generate megaparallax file")
            post_process.post_process(root_dir = str(work_root_dir / dir_item) + "/", dataset_name = scene_name,\
                        replica_camera_file = path_csv_file, replica_center_camera_file = center_csv_file, \
                        original_traj_dir = pose_output_folder, original_center_dir = center_output_folder, \
                        preprocessing_folder = "Preprocessing" + postfix_res + "/", viewer_folder = "Viewer" + postfix_res + "/", \
                        input_image_width = image_width, input_image_height = image_height, input_FPS = FPS)

        except Exception as error:
            print(error)
        

if __name__ == "__main__":
    render_datasets()
