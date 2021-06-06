import os
import sys

# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.abspath(__file__))
dir_root = os.path.dirname(dir_scripts)
dir_utility = os.path.join(dir_root, "utility")
dir_replica360 = os.path.join(dir_root, "replica360")

sys.path.append(dir_scripts)
sys.path.append(dir_root)
sys.path.append(dir_utility)
sys.path.append(dir_replica360)


class ReplicaConfig():
    """The dataset convention 
    """
    # 0) the Replica-Dataset root folder
    replica_data_root_dir = "D:/dataset/replica_v1_0/"
    replica_scene_name_list = ["apartment_0", "frl_apartment_0", "frl_apartment_3", "hotel_0", " office_2", "room_0", "apartment_1", "frl_apartment_1",
                               "frl_apartment_4", "office_0", "office_3", "room_1", "apartment_2", "frl_apartment_2", "frl_apartment_5", "office_1", "office_4", "room_2"]
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
    program_root_dir = "D:/workspace_windows/replica/Replica-Dataset_360/build_msvc/ReplicaSDK/Release/"
    render_panorama_program_filepath = program_root_dir + "ReplicaRendererPanorama.exe"
    render_cubemap_program_filepath = program_root_dir + "ReplicaRendererCubemap.exe"
    renderRGBEnable = True
    renderDepthEnable = False
    renderMotionVectorEnable = True
    renderUnavailableMask = True
