import os

import configuration as config

from utility import flow_estimate, gnomonic_projection
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import projection_cubemap as proj_cm
from utility import flow_warp

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def generate_face_flow():
    """Estimate the flow for each face image.
    The post-fix is dis.
    """
    cubemap_images_src_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    cubemap_images_tar_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_cubemap/")
    src_image_list = []
    tar_image_list = []
    face_image_name_expression = "cubemap_rgb_src_{}.jpg"
    face_flow_name_expression = "cubemap_flow_dis_{}.flo"

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


def test_cubemap_flow_proj(padding_size):
    """Test stitch 6 face optical flow to single ERP flow.
    """
    erp_flow_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    cubemap_flow_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    if not os.path.exists(cubemap_flow_output):
        os.mkdir(cubemap_flow_output)

    # 1) ERP flow to cubemap flow
    erp_flow = flow_io.read_flow_flo(erp_flow_filepath)
    face_flows = proj_cm.erp2cubemap_flow(erp_flow, padding_size)
    for index in range(0, 6):
        cubemap_flow_name = cubemap_flow_output + "cubemap_flo_padding_{}.flo".format(index)
        flow_io.write_flow_flo(face_flows[index], cubemap_flow_name)

        cubemap_flow_vis_name = cubemap_flow_output + "cubemap_flo_padding_{}.jpg".format(index)
        face_flow_vis = flow_vis.flow_to_color(face_flows[index])
        # image_io.image_show(face_flow_vis)
        image_io.image_save(face_flow_vis, cubemap_flow_vis_name)


def test_cubemap_flow_stitch(padding_size):
    """Test project the ERP image flow to 6 face of Cubemap.
    """
    log.info("test_cubemap_flow_stitch")

    # 1) load the flow file to memory
    cubemap_flow_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    face_flows = []
    for index in range(0, 6):
        cubemap_flow_path = cubemap_flow_output + "cubemap_flo_padding_{}.flo".format(index)
        face_flows.append(flow_io.read_flow_flo(cubemap_flow_path))

    # 2) test stitch the cubemap flow. Note enable test 3
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows, erp_flow_height=480, padding_size=padding_size)
    cubemap_stitch_flo = cubemap_flow_output + "cubemap_stitch_flo_padding_test.flo"
    if os.path.exists(cubemap_stitch_flo):
        os.remove(cubemap_stitch_flo)
    flow_io.flow_write(erp_flow_stitch, cubemap_flow_output + "cubemap_stitch_flo_padding_test.flo")

    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    # image_io.image_show(face_flow_vis)
    erp_flow_stitch_name = cubemap_flow_output + "cubemap_stitch_flo_padding_test.jpg"
    image_io.image_save(face_flow_vis, erp_flow_stitch_name)


def test_cubmap_image_proj(padding_size):
    """Project the ERP image the 6 face.
    """
    # 1) erp image to cube map
    erp_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    if not os.path.exists(cubemap_images_output):
        os.mkdir(cubemap_images_output)

    erp_image = image_io.image_read(erp_image_filepath)
    face_images_src = proj_cm.erp2cubemap_image(erp_image, padding_size)

    for index in range(0, len(face_images_src)):
        cubemap_images_name = cubemap_images_output + "cubemap_rgb_src_padding_{}.jpg".format(index)
        image_io.image_save(face_images_src[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


def test_cubmap_image_stitch(padding_size):
    """Test stitch the 6 images to an ERP image.
    """
    cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    # 1) load the 6 image to memory.
    face_images_src = []
    for index in range(0, 6):
        image_path = cubemap_images_output + "cubemap_rgb_src_padding_{}.jpg".format(index)
        face_images_src.append(image_io.image_read(image_path))

    # 2) test stitch the cubemap images
    erp_image_src = proj_cm.cubemap2erp_image(face_images_src, padding_size)
    # image_io.image_show(erp_image_src)
    image_io.image_save(erp_image_src, os.path.join(cubemap_images_output, "0001_rgb_stitch_padding.jpg"))


def test_cubemap_flow_warp():
    """Warp the face image with face flow.
    """
    # 1) warp the cube map image with cube map flow
    cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    rgb_src_filename_exp = "cubemap_rgb_src_padding_{}.jpg"
    flo_filename_exp = "cubemap_flo_padding_{}.flo"
    flo_src_warp_filename_exp = "cubemap_rgb_src_warp_{}.jpg"

    for index in range(0, 6):
        image_path = cubemap_images_output + rgb_src_filename_exp.format(index)
        image_data = image_io.image_read(image_path)
        flow_path = cubemap_images_output + flo_filename_exp.format(index)
        flow_data = flow_io.flow_read(flow_path)

        face_warp_image = flow_warp.warp_forward(image_data, flow_data)
        cubemap_flow_warp_name = cubemap_images_output + flo_src_warp_filename_exp.format(index)
        image_io.image_save(face_warp_image, cubemap_flow_warp_name)
        # image_io.image_show(face_flow_vis)

    # 2) use the ERP optical flow to warp the ERP RGB image
    erp_image_filepath = cubemap_images_output + "../0001_rgb.jpg"
    erp_image = image_io.image_read(erp_image_filepath)
    flow_path = cubemap_images_output + "cubemap_stitch_flo_padding_test.flo"
    flow_data = flow_io.flow_read(flow_path)
    face_warp_image = flow_warp.warp_forward(erp_image, flow_data)
    image_io.image_save(face_warp_image, cubemap_images_output + "0001_rgb_warp.jpg")


if __name__ == "__main__":
    padding_size = 0.0
    # test_cubmap_image_proj(padding_size)
    # test_cubmap_image_stitch(padding_size)

    # 1) test flow stitch and proj
    # generate_face_flow()
    # test_cubemap_flow_proj(padding_size)
    test_cubemap_flow_stitch(padding_size)

    test_cubemap_flow_warp()

    # return 0
    # # image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    # image_path = "/mnt/sda1/workdoc/2020-06-18-360opticalflow/tangent_image_00.png"
    # tangent_image_root = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/output/"
    # gnomonic_projection.sphere2tangent(image_path, tangent_image_root)
    # # gnomonic_projection.tangent2sphere(tangent_image_root, tangent_image_root, [480, 960,  3])
    # test_cubemap_flow_stitch()
