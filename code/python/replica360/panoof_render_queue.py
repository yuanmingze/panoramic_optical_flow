import configuration
from configuration import ReplicaConfig

import json
import os
import subprocess

import cubemap2erp as c2e
import create_rendering_pose
import fs_utility
import depth_io
import flow_io
import image_io
import flow_vis

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


class ReplicaRenderConfig(ReplicaConfig):
    """Replica dataset rendering configuration.
    """
    # 1) the output root folder
    # output_root_dir = "D:/workdata/InstaOmniDepth/replica360/"
    output_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
    output_cubemap_dir = "cubemap/"
    output_pano_dir = "pano/"
    config_json_filename = "config.json"

    def __init__(self):
        # the render and stitch configuration
        self.render_folder_names = []
        self.render_scene_configs = {}
        self.render_scene_pose_files = {}
        self.render_scene_frame_number = {}

    def folder_name_add_postfix(self, folder_postfix):
        log.info("The folder name postfix is : {}".format(folder_postfix))
        for idx in range(len(super().replica_scene_name_list)):
            folder_name = super().replica_scene_name_list[idx] + folder_postfix
            # log.info("Render to folder {}".format(folder_name))
            self.render_folder_names.append(folder_name)


def render_panoramic_datasets(render_configs):
    # 0) genreate the csv camera pose file
    for render_folder_name in render_configs.render_folder_names:
        # load camera path generation configuration
        output_scene_dir = render_configs.output_root_dir + render_folder_name + "/"
        scene_render_config_filepath = output_scene_dir + render_configs.config_json_filename
        with open(scene_render_config_filepath) as json_file:
            config = json.load(json_file)

        if not config["scene_name"] in render_configs.replica_scene_name_list:
            continue
        print("genreate the camera pose csv file for {}".format(config["scene_name"]))

        # genene camera path"
        camera_pose_csv_filepath, _, frame_number = create_rendering_pose.generate_path(output_scene_dir, config)
        render_configs.render_scene_configs[render_folder_name] = config
        render_configs.render_scene_pose_files[render_folder_name] = camera_pose_csv_filepath
        render_configs.render_scene_frame_number[render_folder_name] = frame_number

    # 1) render the cubemap data
    for render_folder_name in render_configs.render_folder_names:
        print("render the cubemap data for {}".format(render_folder_name))
        render_config = render_configs.render_scene_configs[render_folder_name]  # render configuration for each scene
        if not render_config["scene_name"] in render_configs.replica_scene_name_list:
            log.warn("{} is not in replica dataset.".format(render_config["scene_name"]))
            continue

        # check the configuration
        log.info("Rending with {} program.".format(render_config["render_type"]))
        if render_config["render_view"]["center_view"]:
            log.warn("Do not render center viewpoint images.")

        # call the program to render cubemap
        render_args = []
        if render_config["render_type"] == "cubemap":
            render_args.append(render_configs.render_cubemap_program_filepath)
            render_args.append("--imageSize")
            render_args.append(str(render_config["image"]["height"]))
            render_scene_output_dir = render_configs.output_root_dir + render_folder_name + "/" + render_configs.output_cubemap_dir
        elif render_config["render_type"] == "panorama":
            render_args.append(render_configs.render_panorama_program_filepath)
            render_args.append("--imageHeight")
            render_args.append(str(render_config["image"]["height"]))
            render_scene_output_dir = render_configs.output_root_dir + render_folder_name + "/" + render_configs.output_pano_dir
        else:
            log.error("Do not support render method {}".format(render_config["render_type"]))
        fs_utility.dir_make(render_scene_output_dir)

        render_args.append("--data_root")
        render_args.append(render_configs.replica_data_root_dir + render_config["scene_name"] + "/")
        render_args.append("--meshFile")
        render_args.append(render_configs.replica_mesh_file)
        render_args.append("--atlasFolder")
        render_args.append(render_configs.replica_texture_file)
        render_args.append("--mirrorFile")
        render_args.append(render_configs.replica_glass_file)
        render_args.append("--cameraPoseFile")
        render_args.append(render_configs.render_scene_pose_files[render_folder_name])
        render_args.append("--outputDir")
        render_args.append(render_scene_output_dir)
        render_args.append("--texture_exposure")
        render_args.append(str(render_config["render_params"]["texture_exposure"]))
        render_args.append("--texture_gamma")
        render_args.append(str(render_config["render_params"]["texture_gamma"]))
        render_args.append("--texture_saturation")
        render_args.append(str(render_config["render_params"]["texture_saturation"]))
        render_args.append("--renderRGBEnable=" + str(render_configs.renderRGBEnable))
        render_args.append("--renderDepthEnable=" + str(render_configs.renderDepthEnable))
        render_args.append("--renderMotionVectorEnable=" + str(render_configs.renderMotionVectorEnable))

        # run the render program
        print(render_args)
        render_seq_return = subprocess.check_call(render_args)

    # 2) stitch cubemap to panoramic images
    if render_config["render_type"] == "cubemap":
        for render_folder_name in render_configs.render_folder_names:
            pano_output_dir = render_configs.output_root_dir + render_folder_name + "/" + render_configs.output_pano_dir
            print("stitch cubemap to panoramic images for {}".format(render_folder_name))
            render_config = render_configs.render_scene_configs[render_folder_name]
            if not render_config["scene_name"] in render_configs.replica_scene_name_list:
                log.warn("{} is not in replica dataset.".format(render_config["scene_name"]))
                continue

            replica_scene_data_root = render_configs.output_root_dir + render_folder_name + "/" + render_configs.output_cubemap_dir
            fs_utility.dir_make(pano_output_dir)
            # stitch rgb image
            if render_configs.renderRGBEnable:
                log.info("stitch rgb image")
                c2e.stitch_rgb(replica_scene_data_root,
                               pano_output_dir,
                               render_configs.render_scene_frame_number[render_folder_name],
                               render_configs.replica_cubemap_rgb_image_filename_exp,
                               render_configs.replica_pano_rgb_image_filename_exp)
            # stitch depth map
            if render_configs.renderDepthEnable:
                log.info("stitch depth map")
                c2e.stitch_depthmap(replica_scene_data_root,
                                    pano_output_dir,
                                    render_configs.render_scene_frame_number[render_folder_name],
                                    render_configs.replica_cubemap_depthmap_filename_exp,
                                    render_configs.replica_pano_depthmap_filename_exp,
                                    render_configs.replica_pano_depthmap_visual_filename_exp)
            # stitch forward optical flow
            if render_configs.renderMotionVectorEnable:
                log.info("stitch forward optical flow")
                c2e.stitch_opticalflow(replica_scene_data_root,
                                       pano_output_dir,
                                       render_configs.render_scene_frame_number[render_folder_name],
                                       render_configs.replica_cubemap_opticalflow_forward_filename_exp,
                                       render_configs.replica_pano_opticalflow_forward_filename_exp,
                                       render_configs.replica_pano_opticalflow_forward_visual_filename_exp)
                # stitch backward optical flow
                log.info("stitch backward optical flow")
                c2e.stitch_opticalflow(replica_scene_data_root,
                                       pano_output_dir,
                                       render_configs.render_scene_frame_number[render_folder_name],
                                       render_configs.replica_cubemap_opticalflow_backward_filename_exp,
                                       render_configs.replica_pano_opticalflow_backward_filename_exp,
                                       render_configs.replica_pano_opticalflow_backward_visual_filename_exp)

    # 3) generate unavailable mask image & visualize depth map
    for render_folder_name in render_configs.render_folder_names:
        pano_output_dir = render_configs.output_root_dir + render_folder_name + "/" + render_configs.output_pano_dir
        if render_configs.renderUnavailableMask:
            # generate unavailable pixel mask
            if not render_configs.renderDepthEnable:
                log.warn("Need depth map to generate the mask.")
            log.info("enerate unavailable mask image")
            c2e.create_mask(pano_output_dir,
                            pano_output_dir,
                            render_configs.render_scene_frame_number[render_folder_name],
                            render_configs.replica_pano_depthmap_filename_exp,
                            render_configs.replica_pano_mask_filename_exp)

            # visualize depth map and optical flow
            for image_index in range(0, frame_number):
                if image_index % 10 == 0:
                    log.info("Image index: {}".format(image_index))

                erp_depth_filepath = pano_output_dir + render_configs.replica_pano_depthmap_filename_exp.format(image_index)
                erp_depth_visual_filepath = pano_output_dir + render_configs.replica_pano_depthmap_visual_filename_exp.format(image_index)
                erp_depth_data = depth_io.read_dpt(erp_depth_filepath)
                depth_io.depth_visual_save(erp_depth_data, erp_depth_visual_filepath)

        # visualize the optical flow
        if render_configs.renderMotionVectorEnable:
            for image_index in range(0, frame_number):
                if image_index % 10 == 0:
                    log.info("Image index: {}".format(image_index))

                erp_of_filepath = pano_output_dir + render_configs.replica_pano_opticalflow_forward_filename_exp.format(image_index)
                erp_of_visual_filepath = pano_output_dir + render_configs.replica_pano_opticalflow_forward_visual_filename_exp.format(image_index)
                erp_of_data = flow_io.flow_read(erp_of_filepath)
                erp_of_vis = flow_vis.flow_to_color(erp_of_data, min_ratio=0.1, max_ratio=0.9)
                image_io.image_save(erp_of_vis, erp_of_visual_filepath)

                erp_of_filepath = pano_output_dir + render_configs.replica_pano_opticalflow_backward_filename_exp.format(image_index)
                erp_of_visual_filepath = pano_output_dir + render_configs.replica_pano_opticalflow_backward_visual_filename_exp.format(image_index)
                erp_of_data = flow_io.flow_read(erp_of_filepath)
                erp_of_vis = flow_vis.flow_to_color(erp_of_data, min_ratio=0.1, max_ratio=0.9)
                image_io.image_save(erp_of_vis, erp_of_visual_filepath)

    # 3) clean


if __name__ == "__main__":
    render_configs = ReplicaRenderConfig()
    render_configs.folder_name_add_postfix("_rand_1k_0")
    # exit(0)
    # render_configs.render_folder_names = folder_list
    # render_configs.renderMotionVectorEnable = False
    # render_configs.renderDepthEnable = False
    render_panoramic_datasets(render_configs)
