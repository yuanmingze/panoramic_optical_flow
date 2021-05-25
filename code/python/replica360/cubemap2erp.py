
import configuration
import depth_io
import flow_io
import flow_vis
import image_io
import flow_warp
import pointcloud_utils

import projection_cubemap as proj_cm

import os
#  +x, -x, +y, -y, +z, -z
cubemap_face_abbre = ["R", "L", "U", "D", "F", "B"]

# TODO debug the RGB, depth, opticalflow, stitch seam


def cubemap_flow_warp(dataroot_dir, cubemap_rgb_image_filename_exp, cubemap_opticalflow_filename_exp,  cubemap_opticalflow_warp_filename_exp):
    """Warp the face image with face flow.
    """
    for image_index in range(0, 1):
        # 1) warp the cube map image with cube map flow
        for facename_abbr in cubemap_face_abbre:
            image_path = dataroot_dir + cubemap_rgb_image_filename_exp.format(image_index, facename_abbr)
            image_data = image_io.image_read(image_path)
            flow_path = dataroot_dir + cubemap_opticalflow_filename_exp.format(image_index, facename_abbr)
            flow_data = flow_io.flow_read(flow_path)

            face_warp_image = flow_warp.warp_forward(image_data, flow_data)
            cubemap_flow_warp_name = dataroot_dir + cubemap_opticalflow_warp_filename_exp.format(image_index, facename_abbr)
            image_io.image_save(face_warp_image, cubemap_flow_warp_name)
            # image_io.image_show(face_flow_vis)


def erp_depth2pointcloud(dataroot_dir, pano_depthmap_filename_exp, pano_pointcloud_filename_exp):
    """Project the depth map to point clouds.
    """
    for image_index in range(0, 1):
        erp_depth_filepath = dataroot_dir + pano_depthmap_filename_exp.format(image_index)
        erp_pointcloud_filepath = dataroot_dir + pano_pointcloud_filename_exp.format(image_index)
        depth_data = depth_io.read_dpt(erp_depth_filepath)
        # image_io.image_show(depth_data)
        pointcloud_utils.depthmap2pointcloud_erp(depth_data, None, erp_pointcloud_filepath)


def pano_flow_warp(dataroot_dir, pano_rgb_image_filename_exp, pano_opticalflow_filename_exp,  pano_opticalflow_warp_filename_exp):
    """Warp the face image with face flow.
    """
    for image_index in range(0, 1):
        # 1) warp the cube map image with cube map flow
        image_path = dataroot_dir + pano_rgb_image_filename_exp.format(image_index)
        image_data = image_io.image_read(image_path)
        flow_path = dataroot_dir + pano_opticalflow_filename_exp.format(image_index)
        flow_data = flow_io.flow_read(flow_path)

        face_warp_image = flow_warp.warp_forward(image_data, flow_data, True)
        cubemap_flow_warp_name = dataroot_dir + pano_opticalflow_warp_filename_exp.format(image_index)
        image_io.image_save(face_warp_image, cubemap_flow_warp_name)
        # image_io.image_show(face_flow_vis)


def visual_data(data_dir):
    """ Visualize the render result.

    :param data_dir: The path of data folder.
    :type data_dir: str
    """
    counter = 0
    for filename in os.listdir(data_dir):
        counter = counter + 1
        if counter % 10 == 0:
            print(f"{counter} : {filename}")
        if filename.endswith(".dpt"):
            depth_data = depth_io.read_dpt(data_dir + filename)
            depth_io.depth_visual_save(depth_data, data_dir + filename + ".jpg")
        elif filename.endswith(".flo"):
            of_data = flow_io.read_flow_flo(data_dir + filename)
            of_data_vis = flow_vis.flow_to_color(of_data)
            image_io.image_save(of_data_vis, data_dir + filename + ".jpg")


def stitch_rgb(data_dir, cubemap_rgb_image_filename_exp, pano_rgb_image_filename_exp):
    """ Convert the cubemap images to ERP image.
    """
    for image_index in range(0, 2):
        # 1) load the 6 image to memory.
        face_images_src = []
        # for index in range(0, 6):
        for facename_abbr in cubemap_face_abbre:
            image_path = data_dir + cubemap_rgb_image_filename_exp.format(image_index, facename_abbr)
            face_images_src.append(image_io.image_read(image_path))

        # 2) test stitch the cubemap images
        erp_image_data = proj_cm.cubemap2erp_image(face_images_src, 0.0)
        # image_io.image_show(erp_image_src)
        erp_image_filepath = data_dir + pano_rgb_image_filename_exp.format(image_index)
        image_io.image_save(erp_image_data, erp_image_filepath)


def stitch_depthmap(data_dir, cubemap_depthmap_filename_exp, pano_depthmap_filename_exp):
    """ Convert the cubemap depth map to ERP image.

    TODO, convert the perspective depth to radian depth.
    """
    for image_index in range(0, 1):
        # 1) load the 6 image to memory.
        face_depth_list = []
        for facename_abbr in cubemap_face_abbre:
            image_path = data_dir + cubemap_depthmap_filename_exp.format(image_index, facename_abbr)
            print(image_path)
            face_depth_list.append(depth_io.read_dpt(image_path))

        # 2) test stitch the cubemap images
        erp_depth_data = proj_cm.cubemap2erp_depth(face_depth_list, padding_size=0.0)
        erp_depth_filepath = data_dir + pano_depthmap_filename_exp.format(image_index)
        # image_io.image_show(erp_depth_data)
        depth_io.write_dpt(erp_depth_data, erp_depth_filepath)
        depth_io.depth_visual_save(erp_depth_data, erp_depth_filepath + ".jpg")


def stitch_opticalflow(data_dir, cubemap_opticalflow_filename_exp, pano_opticalflow_filename_exp):
    """ Convert the cubemap images to ERP image.
    """
    for image_index in range(0, 1):
        # 1) load the 6 image to memory.
        face_flo_list = []
        for facename_abbr in cubemap_face_abbre:
            image_path = data_dir + cubemap_opticalflow_filename_exp.format(image_index, facename_abbr)
            print(image_path)
            face_flo_list.append(flow_io.read_flow_flo(image_path))

        # 2) test stitch the cubemap images
        erp_depth_data = proj_cm.cubemap2erp_flow(face_flo_list, padding_size=0.0)
        erp_depth_filepath = data_dir + pano_opticalflow_filename_exp.format(image_index)
        flow_io.flow_write(erp_depth_data, erp_depth_filepath)
        erp_depth_vis = flow_vis.flow_to_color(erp_depth_data, [500, 500])
        image_io.image_save(erp_depth_vis, erp_depth_filepath + ".jpg")


if __name__ == "__main__":
    cubemap_rgb_image_filename_exp = "{:04d}_{}_rgb.jpg"
    cubemap_depthmap_filename_exp = "{:04d}_{}_depth.dpt"
    cubemap_opticalflow_forward_filename_exp = "{:04d}_{}_motionvector_forward.flo"
    cubemap_opticalflow_backward_filename_exp = "{:04d}_{}_motionvector_backward.flo"
    cubemap_rgb_forward_warp_filename_exp = "{:04d}_{}_rgb_forward_warp.jpg"

    pano_rgb_image_filename_exp = "{:04d}_rgb.jpg"
    pano_depthmap_filename_exp = "{:04d}_depth.dpt"
    pano_pointcloud_filename_exp = "{:04d}_depth.ply"
    pano_opticalflow_forward_filename_exp = "{:04d}_motionvector_forward.flo"
    pano_opticalflow_backward_filename_exp = "{:04d}_motionvector_backward.flo"
    pano_rgb_forward_warp_filename_exp = "{:04d}_rgb_forward_warp.jpg"

    dataroot_dir = "D:/workdata/opticalflow_data/replic_cubemap/"

    # visual_data(dataroot_dir)
    # stitch_rgb(dataroot_dir, cubemap_rgb_image_filename_exp, pano_rgb_image_filename_exp)
    # stitch_depthmap(dataroot_dir, cubemap_depthmap_filename_exp, pano_depthmap_filename_exp)
    # stitch_opticalflow(dataroot_dir, cubemap_opticalflow_forward_filename_exp, pano_opticalflow_forward_filename_exp)
    # stitch_opticalflow(dataroot_dir, cubemap_opticalflow_backward_filename_exp, pano_opticalflow_backward_filename_exp)
    # cubemap_flow_warp(dataroot_dir, cubemap_rgb_image_filename_exp, cubemap_opticalflow_forward_filename_exp,  cubemap_rgb_forward_warp_filename_exp )
    # pano_flow_warp(dataroot_dir, pano_rgb_image_filename_exp, pano_opticalflow_forward_filename_exp,  pano_rgb_forward_warp_filename_exp)
    erp_depth2pointcloud(dataroot_dir, pano_depthmap_filename_exp, pano_pointcloud_filename_exp)
