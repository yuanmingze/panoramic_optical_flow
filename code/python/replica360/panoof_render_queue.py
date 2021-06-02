import configuration


import json
import os
import subprocess

import cubemap2erp as c2e
import create_rendering_pose
import fs_utility


from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


class ReplicaRenderConfig():
    # 0) the Replica-Dataset root folder
    replica_data_root_dir = "D:/dataset/replica_v1_0/"
    replica_scene_name_list = ["apartment_0", "frl_apartment_0", "frl_apartment_3", "hotel_0", " office_2", "room_0", "apartment_1", "frl_apartment_1",
                               "frl_apartment_4", "office_0", "office_3", "room_1", "apartment_2", "frl_apartment_2", "frl_apartment_5", "office_1", "office_4", "room_2"]
    # original dataset model and texture
    replica_mesh_file = "mesh.ply"
    replica_texture_file = "textures"
    replica_glass_file = "glass.sur"

    replica_cubemap_rgb_image_filename_exp = "{:04d}_{}_rgb.jpg"
    replica_cubemap_depthmap_filename_exp = "{:04d}_{}_depth.dpt"
    replica_cubemap_opticalflow_forward_filename_exp = "{:04d}_{}_motionvector_forward.flo"
    replica_cubemap_opticalflow_backward_filename_exp = "{:04d}_{}_motionvector_backward.flo"

    replica_pano_rgb_image_filename_exp = "{:04d}_rgb_pano.jpg"
    replica_pano_depthmap_filename_exp = "{:04d}_depth_pano.dpt"
    replica_pano_depthmap_visual_filename_exp = "{:04d}_depth_pano_visual.jpg"
    replica_pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    replica_pano_opticalflow_forward_visual_filename_exp = "{:04d}_opticalflow_forward_pano_visual.jpg"
    replica_pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"
    replica_pano_opticalflow_backward_visual_filename_exp = "{:04d}_opticalflow_backward_pano_visual.jpg"
    replica_pano_mask_filename_exp = "{:04d}_mask_pano.png"

    # 1) the output root folder
    output_root_dir = "D:/workdata/opticalflow_data/replic_cubemap/"
    output_cubemap_dir = "cubemap/"
    output_pano_dir =  "pano/"
    config_json_filename = "config.json"

    # 2) data generating programs
    program_root_dir = "D:/workspace_windows/replica/Replica-Dataset_360/build_msvc/ReplicaSDK/Release/"
    render_panorama_program_filepath = program_root_dir + "ReplicaRendererPanorama.exe"
    render_cubemap_program_filepath = program_root_dir + "ReplicaRendererCubemap.exe"
    render_cubemap_imagesize = 640
    renderRGBEnable = False
    renderDepthEnable = False
    renderMotionVectorEnable = False

    # the render and stitch configuration
    render_scene_names = []
    render_scene_configs = {}
    render_scene_pose_files = {}
    render_scene_frame_number = {}


def render_panoramic_datasets(render_configs):
    # 0) genreate the csv camera pose file
    for render_scene_name in render_configs.render_scene_names:
        if not render_scene_name in render_configs.replica_scene_name_list:
            continue
        print("genreate the camera pose csv file for {}".format(render_scene_name))

        # load camera path generation configuration
        output_scene_dir = render_configs.output_root_dir + render_scene_name + "/"
        scene_render_config_filepath = output_scene_dir + render_configs.config_json_filename
        with open(scene_render_config_filepath) as json_file:
            config = json.load(json_file)

        # genene camera path"
        camera_pose_csv_filepath, _, frame_number = create_rendering_pose.generate_path(output_scene_dir, config)
        render_configs.render_scene_configs[render_scene_name] = config
        render_configs.render_scene_pose_files[render_scene_name] = camera_pose_csv_filepath
        render_configs.render_scene_frame_number[render_scene_name] = frame_number

    # 1) render the cubemap data
    for render_scene_name in render_configs.render_scene_names:
        if not render_scene_name in render_configs.replica_scene_name_list:
            continue
        print("render the cubemap data for {}".format(render_scene_name))

        render_config = render_configs.render_scene_configs[render_scene_name]

        # check the configurtion
        if render_config["render_type"] == "panorama":
            log.warn("Do not support render panorama images.")
        if render_config["render_view"]["center_view"]:
            log.warn("Do not render center viewpoint images.")

        render_scene_output_dir = render_configs.output_root_dir + render_scene_name + "/" + render_configs.output_cubemap_dir
        fs_utility.dir_make(render_scene_output_dir)

        # call the program to render cubemap
        render_args = [render_configs.render_cubemap_program_filepath]
        render_args.append("--data_root")
        render_args.append(render_configs.replica_data_root_dir + render_scene_name + "/")
        render_args.append("--meshFile")
        render_args.append(render_configs.replica_mesh_file)
        render_args.append("--atlasFolder")
        render_args.append(render_configs.replica_texture_file)
        render_args.append("--mirrorFile")
        render_args.append(render_configs.replica_glass_file)
        render_args.append("--cameraPoseFile")
        render_args.append(render_configs.render_scene_pose_files[render_scene_name])
        render_args.append("--outputDir")
        render_args.append(render_scene_output_dir)
        render_args.append("--imageSize")
        render_args.append(str(render_configs.render_cubemap_imagesize))
        render_args.append("--renderRGBEnable")
        render_args.append(str(render_configs.renderRGBEnable))
        render_args.append("--renderDepthEnable")
        render_args.append(str(render_configs.renderDepthEnable))
        render_args.append("--renderMotionVectorEnable")
        render_args.append(str(render_configs.renderMotionVectorEnable))

        # run the render program
        # render_seq_return = subprocess.check_call(render_args)

    # 2) stitch cubemap to panoramic images
    for render_scene_name in render_configs.render_scene_names:
        if not render_scene_name in render_configs.replica_scene_name_list:
            continue
        print("stitch cubemap to panoramic images for {}".format(render_scene_name))

        replica_scene_data_root = render_configs.output_root_dir + render_scene_name + "/" + render_configs.output_cubemap_dir
        pano_output_dir = render_configs.output_root_dir + render_scene_name + "/" + render_configs.output_pano_dir
        fs_utility.dir_make(pano_output_dir)
        # stitch rgb image
        log.info("stitch rgb image")
        c2e.stitch_rgb(replica_scene_data_root,
                       pano_output_dir,
                       render_configs.render_scene_frame_number[render_scene_name],
                       render_configs.replica_cubemap_rgb_image_filename_exp,
                       render_configs.replica_pano_rgb_image_filename_exp)
        # stitch depth map
        log.info("stitch depth map")
        c2e.stitch_depthmap(replica_scene_data_root,
                            pano_output_dir,
                            render_configs.render_scene_frame_number[render_scene_name],
                            render_configs.replica_cubemap_depthmap_filename_exp,
                            render_configs.replica_pano_depthmap_filename_exp,
                            render_configs.replica_pano_depthmap_visual_filename_exp)
        # stitch forward optical flow
        log.info("stitch forward optical flow")
        c2e.stitch_opticalflow(replica_scene_data_root,
                               pano_output_dir,
                               render_configs.render_scene_frame_number[render_scene_name],
                               render_configs.replica_cubemap_opticalflow_forward_filename_exp,
                               render_configs.replica_pano_opticalflow_forward_filename_exp,
                               render_configs.replica_pano_opticalflow_forward_visual_filename_exp)
        # stitch backward optical flow
        log.info("stitch backward optical flow")
        c2e.stitch_opticalflow(replica_scene_data_root,
                               pano_output_dir,
                               render_configs.render_scene_frame_number[render_scene_name],
                               render_configs.replica_cubemap_opticalflow_backward_filename_exp,
                               render_configs.replica_pano_opticalflow_backward_filename_exp,
                               render_configs.replica_pano_opticalflow_backward_visual_filename_exp)
        # generate unavailable mask image
        log.info("enerate unavailable mask image")
        c2e.create_mask(pano_output_dir,
                        pano_output_dir,
                        render_configs.render_scene_frame_number[render_scene_name],
                        render_configs.replica_pano_depthmap_filename_exp,
                        render_configs.replica_pano_mask_filename_exp)

    # 3) clean


if __name__ == "__main__":
    render_config = ReplicaRenderConfig()
    render_config.render_scene_names = ["office_0"]
    render_panoramic_datasets(render_config)
