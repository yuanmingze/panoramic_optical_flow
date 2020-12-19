import os

import configuration as config

from utility import flow_estimate, gnomonic_projection
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import projection_cubemap as proj_cm


def test_cubemap_flow_proj():
    """Test stitch 6 face optical flow to single ERP flow.
    """
    erp_flow_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    cubemap_flow_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    if not os.path.exists(cubemap_flow_output):
        os.mkdir(cubemap_flow_output)

    # 1) ERP flow to cubemap flow
    erp_flow = flow_io.read_flow_flo(erp_flow_filepath)
    face_flows = proj_cm.erp2cubemap_flow(erp_flow)
    for index in range(0, len(face_flows)):
        cubemap_flow_name = cubemap_flow_output + "cubemap_flo_{}.jpg".format(index)
        cubemap_flow_image_name = cubemap_flow_output + "cubemap_flo_{}.flo".format(index) 
        flow_io.flow_write(face_flows[index], cubemap_flow_image_name)
        face_flow_vis = flow_vis.flow_to_color(face_flows[index])
        # image_io.image_show(face_flow_vis)
        image_io.image_save(face_flow_vis, cubemap_flow_name)


def test_cubemap_flow_stitch():
    """Test project the ERP image flow to 6 face of Cubemap.
    """
    # 1) load the flow file to memory
    cubemap_flow_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    face_flows = []
    for index in range(0, 6):
        cubemap_flow_path = cubemap_flow_output + "cubemap_flow_dis_{}.flo".format(index)
        face_flows.append(flow_io.read_flow_flo(cubemap_flow_path))

    # 2) test stitch the cubemap flow. Note enable test 3
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows)
    erp_flow_stitch_name = cubemap_flow_output + "cubemap_stitch_flo_test.jpg"
    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    # image_io.image_show(face_flow_vis)
    image_io.image_save(face_flow_vis, erp_flow_stitch_name)


def test_cubmap_image_proj():
    """Project the ERP image the 6 face.
    """
    # 1) erp image to cube map
    erp_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    if not os.path.exists(cubemap_images_output):
        os.mkdir(cubemap_images_output)

    erp_image = image_io.image_read(erp_image_filepath)
    face_images_src = proj_cm.erp2cubemap_image(erp_image)

    for index in range(0, len(face_images_src)):
        cubemap_images_name = cubemap_images_output + "cubemap_rgb_src_{}.jpg".format(index)
        image_io.image_save(face_images_src[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


def test_cubmap_image_stitch():
    """Test stitch the 6 images to an ERP image.
    """
    cubemap_images_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_cubemap/")
    # 1) load the 6 image to memory.
    face_images_src = []
    for index in range(0, 6):
        image_path = cubemap_images_output + "cubemap_rgb_src_{}.jpg".format(index)
        image_path = os.path.join(config.TEST_data_root_dir, )
        face_images_src.append(image_io.image_read(image_path))

    # 2) test stitch the cubemap images
    erp_image_src = proj_cm.cubemap2erp_image(face_images_src)
    image_io.image_show(erp_image_src)
    image_io.image_save(erp_image_src, os.path.join(cubemap_images_output, "0001_rgb_stitch.jpg"))


def test_generage_cubic_ply():
    # 1) output cubemap ply mesh
    cubemap_ply_filepath = "../../data/cubemap_points.ply"
    proj_cm.generage_cubic_ply(cubemap_ply_filepath)


def test_cubemap_flow_warp():
    """Warp the face image with face flow.
    """


    # # 2) erp image to cube map
    # erp_image_filepath = os.path.join(config.TEST_data_root_dir, "/replica_360/apartment_0/0001_rgb.jpg")
    # cubemap_images_output = config.TEST_data_root_dir
    # erp_image = image_io.image_read(erp_image_filepath)
    # face_images_src = proj_cm.erp2cubemap_image(erp_image)
    # for index in range(0, len(face_images_src)):
    #     cubemap_images_name = cubemap_images_output + "cubemap_rgb_src_{}.jpg".format(index)
    #     image_io.image_save(face_images_src[index], cubemap_images_name)
    #     # image_io.image_show(face_images[0])

    # erp_image_filepath = "../../data/replica_360/apartment_0/0002_rgb.jpg"
    # cubemap_images_output = "../../data/"
    # erp_image = image_io.image_read(erp_image_filepath)
    # face_images_tar = erp2cubemap_image(erp_image)
    # for index in range(0, len(face_images_tar)):
    #     cubemap_images_name = cubemap_images_output + "cubemap_rgb_tar_{}.jpg".format(index)
    #     image_io.image_save(face_images_tar[index], cubemap_images_name)
    #     # image_io.image_show(face_images[0])

    # 3) erp flow to cube map
    # erp_flow_filepath = "../../data/replica_360/hotel_0/0001_opticalflow_forward.flo"
    erp_flow_filepath = os.path.join(config.TEST_data_root_dir,"/replica_360/apartment_0/0001_opticalflow_forward.flo")
    cubemap_flow_output = "../../data/"
    erp_flow = flow_io.readFlowFile(erp_flow_filepath)
    face_flows = proj_cm.erp2cubemap_flow(erp_flow)
    for index in range(0, len(face_flows)):
        cubemap_flow_name = cubemap_flow_output + "cubemap_flo_{}.jpg".format(index)
        face_flow_vis = flow_vis.flow_to_color(face_flows[index])
        image_io.image_save(face_flow_vis, cubemap_flow_name)
        # image_io.image_show(face_flow_vis)

    # # 4) warp the cube map image with cube map flow
    # cubemap_flow_warp_output = "../../data/"
    # for index in range(0, len(face_flows)):
    #     cubemap_flow_warp_name = cubemap_flow_output + "cubemap_warp_{}.jpg".format(index)
    #     face_warp_image = flow_warp.warp_forward(face_images_src[index], face_flows[index])
    #     image_io.image_save(face_warp_image, cubemap_flow_warp_name)
    #     # image_io.image_show(face_flow_vis)

    # 6) test stitch the cubemap flow. Note enable test 3
    erp_flow_stitch = proj_cm.cubemap2erp_flow(face_flows)
    erp_flow_stitch_name = cubemap_flow_output + "cubemap_stitch_flo_test.jpg"
    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch)
    # image_io.image_show(face_flow_vis)
    image_io.image_save(face_flow_vis, erp_flow_stitch_name)


if __name__ == "__main__":
    # test_cubmap_image_proj()
    # generate_face_flow()
    # test_cubemap_image()
    test_cubemap_flow_proj()
    # return 0
    # # image_path = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/replica_360/hotel_0/0001_rgb.jpg"
    # image_path = "/mnt/sda1/workdoc/2020-06-18-360opticalflow/tangent_image_00.png"
    # tangent_image_root = "/mnt/sda1/workspace_windows/panoramic_optical_flow/data/output/"
    # gnomonic_projection.sphere2tangent(image_path, tangent_image_root)
    # # gnomonic_projection.tangent2sphere(tangent_image_root, tangent_image_root, [480, 960,  3])
    # test_cubemap_flow_stitch()
