import re
import sys
import pathlib
import platform

INDEX_MAX = sys.maxsize

class ReplicaConfig():
    """The dataset convention 
    """
    # 0) the Replica-Dataset root folder
    if platform.system() == "Windows":
        replica_data_root_dir = "D:/dataset/replica_v1_0/"
    elif platform.system() == "Linux":
        replica_data_root_dir = "/mnt/sda1/dataset/replica_v1_0/"

    # # all dataset list
    # "apartment_0",
    # "apartment_1",
    # "apartment_2",
    # "frl_apartment_0",
    # "frl_apartment_1",
    # "frl_apartment_2",
    # "frl_apartment_3",
    # "frl_apartment_4",
    # "frl_apartment_5",
    # "hotel_0",
    # "office_0",
    # "office_1",
    # "office_2",
    # "office_3",
    # "office_4",
    # "room_0",
    # "room_1",
    # "room_2"

    replica_scene_name_list = [
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
        "room_1",
    ]

    # original dataset model and texture
    replica_mesh_file = "mesh.ply"
    replica_texture_file = "textures"
    replica_glass_file = "glass.sur"

    # cubemap filename expression
    replica_cubemap_rgb_image_filename_exp = "{:04d}_{}_rgb.jpg"
    replica_cubemap_depthmap_filename_exp = "{:04d}_{}_depth.dpt"
    replica_cubemap_opticalflow_forward_filename_exp = "{:04d}_{}_motionvector_forward.flo"
    replica_cubemap_opticalflow_backward_filename_exp = "{:04d}_{}_motionvector_backward.flo"

    replica_cubemap_rgb_froward_of_forwardwarp_filename_exp = "{:04d}_{}_motionvector_forward_rgb_forwardwarp.jpg"
    replica_cubemap_rgb_backward_of_forwardwarp_filename_exp = "{:04d}_{}_motionvector_backward_rgb_forwardwarp.jpg"

    # panoramic filename expresion
    replica_pano_rgb_image_filename_exp = "{:04d}_rgb_pano.jpg"
    replica_pano_depthmap_filename_exp = "{:04d}_depth_pano.dpt"
    replica_pano_depthmap_visual_filename_exp = "{:04d}_depth_pano_visual.jpg"
    replica_pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    replica_pano_opticalflow_forward_visual_filename_exp = "{:04d}_opticalflow_forward_pano_visual.jpg"
    replica_pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"
    replica_pano_opticalflow_backward_visual_filename_exp = "{:04d}_opticalflow_backward_pano_visual.jpg"
    replica_pano_mask_filename_exp = "{:04d}_mask_pano.png"

    replica_pano_rgb_froward_of_forwardwarp_filename_exp = "{:04d}_opticalflow_forward_rgb_forwardwarp.jpg"
    replica_pano_rgb_backward_of_forwardwarp_filename_exp = "{:04d}_opticalflow_backward_rgb_forwardwarp.jpg"

    # 2) data generating programs
    if platform.system() == "Windows":
        program_root_dir = "D:/workspace_windows/replica/Replica-Dataset_360/build_msvc/ReplicaSDK/Release/"
        render_panorama_program_filepath = program_root_dir + "ReplicaRendererPanorama.exe"
        render_cubemap_program_filepath = program_root_dir + "ReplicaRendererCubemap.exe"
    elif platform.system() == "Linux":
        program_root_dir = "/mnt/sda1/workspace_windows/replica/Rendering360OpticalFlow/build_linux/ReplicaSDK/"
        render_panorama_program_filepath = program_root_dir + "ReplicaRendererPanorama"
        render_cubemap_program_filepath = program_root_dir + "ReplicaRendererCubemap"
    renderRGBEnable = True
    renderDepthEnable = True
    renderMotionVectorEnable = True
    renderUnavailableMask = True


class ReplicaPanoDataset(ReplicaConfig):

    if sys.platform == 'win32':
        pano_dataset_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
        result_output_dir = "D:/workspace_windows/panoramic_optical_flow/data/replica_result/"
    elif sys.platform == 'linux':
        pano_dataset_root_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/"
        result_output_dir = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_result/"

    pano_data_dir = "pano/"

    pano_output_dir = "result/"
    pano_output_csv = "result_replica.csv"

    padding_size = None

    # circle data
    dataset_circ_dirlist = [
        "apartment_0_circ_1k_0",
        "apartment_1_circ_1k_0",
        # "apartment_2_circ_1k_0",
        "frl_apartment_0_circ_1k_0",
        # "frl_apartment_1_circ_1k_0",
        "frl_apartment_2_circ_1k_0",
        "frl_apartment_3_circ_1k_0",
        # "frl_apartment_4_circ_1k_0",
        # "frl_apartment_5_circ_1k_0",
        "hotel_0_circ_1k_0",
        "office_0_circ_1k_0",
        "office_1_circ_1k_0",
        "office_2_circ_1k_0",
        "office_3_circ_1k_0",
        "office_4_circ_1k_0",
        "room_0_circ_1k_0",
        "room_1_circ_1k_0",
        # "room_2_circ_1k_0",
    ]
    circle_start_idx = 4
    circle_end_idx = 6

    # line data
    dataset_line_dirlist = [
        "apartment_0_line_1k_0",
        "apartment_1_line_1k_0",
        # "apartment_2_line_1k_0",
        "frl_apartment_0_line_1k_0",
        # "frl_apartment_1_line_1k_0",
        "frl_apartment_2_line_1k_0",
        "frl_apartment_3_line_1k_0",
        # "frl_apartment_4_grid_1k_0",
        # "frl_apartment_5_grid_1k_0",
        "hotel_0_line_1k_0",
        "office_0_line_1k_0",
        "office_1_line_1k_0",
        "office_2_line_1k_0",
        "office_3_line_1k_0",
        "office_4_line_1k_0",
        "room_0_line_1k_0",
        "room_1_line_1k_0",
        # "room_2_grid_1k_0",
    ]
    line_start_idx = 4
    line_end_idx = 6

    # random data
    dataset_rand_dirlist = [
        "apartment_0_rand_1k_0",
        "apartment_1_rand_1k_0",
        # "apartment_2_rand_1k_0",
        "frl_apartment_0_rand_1k_0",
        # "frl_apartment_1_rand_1k_0",
        "frl_apartment_2_rand_1k_0",
        "frl_apartment_3_rand_1k_0",
        # "frl_apartment_4_rand_1k_0",
        # "frl_apartment_5_rand_1k_0",
        "hotel_0_rand_1k_0",
        "office_0_rand_1k_0",
        "office_1_rand_1k_0",
        "office_2_rand_1k_0",
        "office_3_rand_1k_0",
        "office_4_rand_1k_0",
        "room_0_rand_1k_0",
        "room_1_rand_1k_0",
        # "room_2_rand_1k_0",
    ]
    rand_start_idx = 4
    rand_end_idx = 6


class OmniPhotoDataset():

    if sys.platform == 'win32':
        pano_dataset_root_dir = "D:/workdata/omniphotos_bmvc_2021/"
        result_output_dir = "D:/workspace_windows/panoramic_optical_flow/data/omniphotos_result/"
    elif sys.platform == 'linux':
        pano_dataset_root_dir = "/mnt/sda1/workdata/omniphotos_bmvc_2021/"
        result_output_dir = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/omniphotos_result/"

    pano_data_dir = "pano/"

    pano_output_dir = "result/"
    pano_output_csv = "result_omniphotos.csv"

    pano_image_height = 640

    # file name expression
    pano_rgb_image_filename_exp = "panoramic-{:04d}.jpg"
    pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    pano_opticalflow_forward_visual_filename_exp = "{:04d}_opticalflow_forward_pano_visual.jpg"
    pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"
    pano_opticalflow_backward_visual_filename_exp = "{:04d}_opticalflow_backward_pano_visual.jpg"

    # circle data
    dataset_circ_dirlist = [
        "Ballintoy",
        "BathAbbey2",
        "BathParadeGardens",
        "BeihaiPark",
        "Coast",
        "DublinShip1",
        "OsakaTemple6",
        "SecretGarden1",
        "Wulongting",
    ]


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
