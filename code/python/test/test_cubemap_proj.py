import os

import configuration as config

from utility import flow_estimate, flow_post_proc, gnomonic_projection
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import projection_cubemap as proj_cm
from utility import flow_warp
from utility import depth_io

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def compute_erp_flow(src_image_path, target_image_path, optical_flow_output_path):
    """compute the optical flow with ERP image.
    """
    src_image = image_io.image_read(src_image_path)
    tar_image = image_io.image_read(target_image_path)

    face_flow = flow_estimate.DIS(src_image, tar_image)
    face_flow = flow_post_proc.of_ph2pano(face_flow)
    flow_io.flow_write(face_flow, optical_flow_output_path)
    face_flow_vis = flow_vis.flow_to_color(face_flow)
    image_io.image_save(face_flow_vis, optical_flow_output_path + ".jpg")


def generate_face_forward_flow(cubemap_images_src_output, cubemap_images_tar_output, face_image_name_expression, face_flow_name_expression):
    """Estimate the flow for each face image. output to cubemap source images' folder.
    The post-fix is dis.
    """
    src_image_list = []
    tar_image_list = []

    # 1) load images
    for index in range(0, 6):
        face_image_name = face_image_name_expression.format(index)
        src_image_list.append(image_io.image_read(os.path.join(cubemap_images_src_output, face_image_name)))
        tar_image_list.append(image_io.image_read(os.path.join(cubemap_images_tar_output, face_image_name)))

    # 2) estimate them DIS optical flow
    for index in range(0, 6):
        face_flow = flow_estimate.DIS(src_image_list[index], tar_image_list[index])
        face_flow_path = os.path.join(cubemap_images_src_output, face_flow_name_expression.format(index))
        flow_io.flow_write(face_flow, face_flow_path)
        face_flow_vis = flow_vis.flow_to_color(face_flow)
        image_io.image_save(face_flow_vis, face_flow_path + ".jpg")


def test_cubemap_flow_proj(padding_size, erp_flow_filepath, cubemap_flow_output, face_gt_flow_padding_name_expression, face_image_padding_name_expression, face_image_size):
    """Test stitch 6 face optical flow to single ERP flow.
    """
    if not os.path.exists(cubemap_flow_output):
        os.mkdir(cubemap_flow_output)

    # 1) ERP flow to cubemap flow
    erp_flow = flow_io.read_flow_flo(erp_flow_filepath)
    face_flows = proj_cm.erp2cubemap_flow(erp_flow, padding_size, face_image_size)
    for index in range(0, 6):
        cubemap_flow_name = cubemap_flow_output + face_gt_flow_padding_name_expression.format(index)
        flow_io.write_flow_flo(face_flows[index], cubemap_flow_name)

        cubemap_flow_vis_name = cubemap_flow_output + face_gt_flow_padding_name_expression.format(index) + ".jpg"
        face_flow_vis = flow_vis.flow_to_color(face_flows[index])
        # image_io.image_show(face_flow_vis)
        image_io.image_save(face_flow_vis, cubemap_flow_vis_name)


def test_cubemap_flow_stitch(padding_size, cubemap_flow_output, flow_filename_expression, cubemap_stitch_flo):
    """Test project the ERP image flow to 6 face of Cubemap.
    """
    log.info("test_cubemap_flow_stitch")
    # 1) load the flow file to memory
    face_flows = []
    for index in range(0, 6):
        cubemap_flow_path = cubemap_flow_output + flow_filename_expression.format(index)
        face_flows.append(flow_io.read_flow_flo(cubemap_flow_path))

    # 2) test stitch the cubemap flow. Note enable test 3
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows, erp_flow_height=480, padding_size=padding_size)
    if os.path.exists(cubemap_stitch_flo):
        os.remove(cubemap_stitch_flo)
    flow_io.flow_write(erp_flow_stitch, cubemap_stitch_flo)

    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    # image_io.image_show(face_flow_vis)
    erp_flow_stitch_name = cubemap_stitch_flo + ".jpg"
    image_io.image_save(face_flow_vis, erp_flow_stitch_name)


def test_cubmap_image_proj(padding_size, erp_image_filepath, cubemap_images_output_folder, face_image_name_expression, face_image_size):
    """Project the ERP image the 6 face.
    """
    # 1) erp image to cube map
    if not os.path.exists(cubemap_images_output_folder):
        os.mkdir(cubemap_images_output_folder)

    erp_image = image_io.image_read(erp_image_filepath)
    face_images_src = proj_cm.erp2cubemap_image(erp_image, padding_size, face_image_size)

    for index in range(0, len(face_images_src)):
        cubemap_images_name = cubemap_images_output_folder + face_image_name_expression.format(index)
        image_io.image_save(face_images_src[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


def test_cubmap_image_stitch(padding_size, erp_image_filepath, cubemap_images_output_folder, face_image_name_expression):
    """Test stitch the 6 images to an ERP image.
    """
    # 1) load the 6 image to memory.
    face_images_src = []
    for index in range(0, 6):
        image_path = cubemap_images_output_folder + face_image_name_expression.format(index)
        face_images_src.append(image_io.image_read(image_path))

    # 2) test stitch the cubemap images
    erp_image_src = proj_cm.cubemap2erp_image(face_images_src, padding_size)
    # image_io.image_show(erp_image_src)
    image_io.image_save(erp_image_src, erp_image_filepath + "_stitch.jpg")


def test_cubemap_flow_warp(cubemap_images_output,rgb_src_filename_exp, flo_filename_exp,  flo_src_warp_filename_exp ):
    """Warp the face image with face flow.
    """
    # 1) warp the cube map image with cube map flow
    for index in range(0, 6):
        image_path = cubemap_images_output + rgb_src_filename_exp.format(index)
        image_data = image_io.image_read(image_path)
        flow_path = cubemap_images_output + flo_filename_exp.format(index)
        flow_data = flow_io.flow_read(flow_path)

        face_warp_image = flow_warp.warp_forward(image_data, flow_data, True)
        cubemap_flow_warp_name = cubemap_images_output + flo_src_warp_filename_exp.format(index)
        image_io.image_save(face_warp_image, cubemap_flow_warp_name)
        # image_io.image_show(face_flow_vis)


def test_erp_flow_warp(erp_image_filepath, erp_flow_filepath, erp_image_warped_filepath):
    """[summary]
    """
    # # 1) warp the cube map image with cube map flow
    # cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    # rgb_src_filename_exp = "cubemap_rgb_src_padding_{}.jpg"
    # flo_filename_exp = "cubemap_flo_padding_{}.flo"
    # flo_src_warp_filename_exp = "cubemap_rgb_src_warp_{}.jpg"

    # use the ERP optical flow to warp the ERP RGB image
    erp_image = image_io.image_read(erp_image_filepath)
    flow_data = flow_io.flow_read(erp_flow_filepath)
    face_warp_image = flow_warp.warp_forward(erp_image, flow_data, True)
    image_io.image_save(face_warp_image, erp_image_warped_filepath)


def test_cubemap_flow_stitch_dis(padding_size, cubemap_flow_dir, face_flow_name_expression, cubemap_stitch_flo, src_erp_image, tar_erp_image):
    """Test stitch the dis's optical flow to ERP optical flow with weight padding.
    """
    log.info("{} stitch dis optical flow.".format(__name__))

    # 1) load the flow file to memory
    face_flows = []
    for index in range(0, 6):
        cubemap_flow_path = cubemap_flow_dir + face_flow_name_expression.format(index)
        face_flows.append(flow_io.read_flow_flo(cubemap_flow_path))

    # 2) test stitch the cubemap flow.
    image_erp_src = image_io.image_read(src_erp_image)
    image_erp_tar = image_io.image_read(tar_erp_image)
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows, erp_flow_height=480, padding_size=padding_size, image_erp_src=image_erp_src, image_erp_tar=image_erp_tar)
    if os.path.exists(cubemap_stitch_flo):
        os.remove(cubemap_stitch_flo)
        log.warn("remove exist flow file {}".format(cubemap_stitch_flo))
    flow_io.flow_write(erp_flow_stitch, cubemap_stitch_flo)
    # erp_flow_stitch = flow_post_proc.of_ph2pano(erp_flow_stitch)  # tranform to the second class optica flow.
    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    # image_io.image_show(face_flow_vis)
    image_io.image_save(face_flow_vis, cubemap_stitch_flo + ".jpg")


def test_cubemap_depth_proj_stitch():
    """Test stitch cubemap depht map to ERP depth map.
    """
    # 1) project the ERP RGB image to ERP RGB Image & estimate each face image's depth
    padding_size = 0.2
    erp_rgb_path = os.path.join(config.TEST_data_root_dir, "omniphotos/OsakaTemple6/OsakaTemple6-00.jpg")
    erp_rgb_data = image_io.image_read(erp_rgb_path)
    cubemap_rgb_list = proj_cm.erp2cubemap_image(erp_rgb_data, padding_size)

    # 2) estimate each face image's depth
    for index in range(0, len(cubemap_rgb_list)):
        cubemap_faceimage_path = os.path.join(config.TEST_data_root_dir, "omniphotos/OsakaTemple6/OsakaTemple6-cubemap-{}.jpg".format(index))
        image_io.image_save(cubemap_rgb_list[index], cubemap_faceimage_path)

    # 3) stitch the cubemap depth map to ERP image.
    erp_depth_path = os.path.join(config.TEST_data_root_dir, "omniphotos/OsakaTemple6/OsakaTemple6-00.pfm")
    face_depthmap_list = []
    for index in range(0, len(cubemap_rgb_list)):
        cubemap_facedepth_path = os.path.join(config.TEST_data_root_dir, "omniphotos/OsakaTemple6/OsakaTemple6-cubemap-{}.pfm".format(index))
        face_depthmap, _ = depth_io.read_pfm(cubemap_facedepth_path)
        face_depthmap_list.append(face_depthmap)

    # 4) stitch the depth map
    erp_depth_data = proj_cm.cubemap2erp_depth(face_depthmap_list, padding_size=padding_size)
    depth_io.depth_visual_save(erp_depth_data, erp_depth_path + ".jpg")
    depth_io.write_pfm(erp_depth_path, erp_depth_data, scale=1)


if __name__ == "__main__":
    # test_cubemap_depth_proj_stitch()
    # exit()

    padding_size = 0.1

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")
    erp_flow_gt_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")

    cubemap_src_images_output_folder = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    cubemap_tar_images_output_folder = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_cubemap/")

    cubemap_stitch_flo_filename = cubemap_src_images_output_folder + "cubemap_flow_dis_padding_stitch.flo"
    cubemap_stitch_gt_flo_filename = cubemap_src_images_output_folder + "cubemap_flow_gt_padding_stitch.flo"
    erp_image_warped_filepath = cubemap_src_images_output_folder + "0001_rgb_warp_dis_padding.jpg"

    face_image_padding_name_expression = "cubemap_rgb_padding_{}.jpg"
    face_gt_flow_padding_name_expression = "cubemap_flow_gt_padding_{}.flo"
    face_dis_flow_padding_name_expression = "cubemap_flow_dis_padding_{}.flo"
    face_image_flo_warp_name_expression = "cubemap_rgb_src_warp_{}.jpg"

    face_image_size = 400

    # 0) test rgb image project and stitch
    test_cubmap_image_proj(padding_size, erp_src_image_filepath, cubemap_src_images_output_folder, face_image_padding_name_expression, face_image_size)
    test_cubmap_image_stitch(padding_size, erp_src_image_filepath, cubemap_src_images_output_folder, face_image_padding_name_expression)

    # 1) test on ground truth optical flow
    test_cubemap_flow_proj(padding_size, erp_flow_gt_filepath, cubemap_src_images_output_folder, face_gt_flow_padding_name_expression, face_image_padding_name_expression, face_image_size)
    test_cubemap_flow_stitch(padding_size, cubemap_src_images_output_folder,  face_gt_flow_padding_name_expression, cubemap_stitch_gt_flo_filename)
    test_cubemap_flow_warp(cubemap_src_images_output_folder,face_image_padding_name_expression, face_gt_flow_padding_name_expression,  face_image_flo_warp_name_expression)

    # 2) test on DIS estimated optical flow with padding
    # project the source image to 6 face
    log.info("project the source image to 6 face")
    test_cubmap_image_proj(padding_size, erp_src_image_filepath, cubemap_src_images_output_folder, face_image_padding_name_expression,face_image_size)
    # project the target image to 6 face
    log.info("project the target image to 6 face")
    test_cubmap_image_proj(padding_size, erp_tar_image_filepath, cubemap_tar_images_output_folder, face_image_padding_name_expression,face_image_size)
    # compute the forward flow
    log.info("compute the forward flow")
    generate_face_forward_flow(cubemap_src_images_output_folder, cubemap_tar_images_output_folder, face_image_padding_name_expression, face_dis_flow_padding_name_expression)
    # stitch all faces
    log.info("stitch all faces")
    test_cubemap_flow_stitch_dis(padding_size, cubemap_src_images_output_folder, face_dis_flow_padding_name_expression, cubemap_stitch_flo_filename, erp_src_image_filepath, erp_tar_image_filepath)
    log.info("use stitched dis optical flow to wrap the src image")
    test_erp_flow_warp(erp_src_image_filepath, cubemap_stitch_flo_filename, erp_image_warped_filepath)

    # 3) test ERP DIS
    # compute_erp_flow(erp_src_image_filepath, erp_tar_image_filepath, erp_flow_path)
