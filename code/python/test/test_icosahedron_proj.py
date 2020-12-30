import os

import configuration as config

from utility import image_io
from utility import projection_icosahedron as proj_ico


def test_ico_parameters(padding_size):
    """
    Check the icosahedron's paramters.
    """
    for index in range(0,20):
        ico_parameter = proj_ico.get_icosahedron_parameters(index, padding_size)
        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(index, ico_parameter["triangle_points_tangent"], ico_parameter["triangle_points_sph"]))

        # plot the parameter
        tangent_point = ico_parameter["tangent_points"]
        triangle_points_tangent = ico_parameter["triangle_points_tangent"]
        triangle_points_sph = ico_parameter["triangle_points_sph"]
        availied_ERP_area = ico_parameter["availied_ERP_area"]  

        # TODO plot the points




def test_ico_image_proj(erp_image_filepath, ico_images_expression, ico_image_output, tangent_image_size, padding_size):
    """
    Project the ERP image to 20 faces flow.
    """
    # test
    erp_image = image_io.image_read(erp_image_filepath)

    tangent_image_list = proj_ico.erp2ico_image(erp_image, tangent_image_size, padding_size)
    for index in range(0, len(tangent_image_list)):
        cubemap_images_name = ico_image_output + ico_images_expression.format(index)
        image_io.image_save(tangent_image_list[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


def test_ico_image_stitch(input_folder_path, file_expression, erp_src_image_stitch_filepath, erp_image_height=960):
    """
    test stitch the icosahedron's image face.
    """
    # load the 20 tangnet images
    tangnet_images_list = []
    for index in range(0, 20):
        tangnet_images_name = input_folder_path + file_expression.format(index)
        tangnet_images_list.append(image_io.image_read(tangnet_images_name))

    # stitch image
    erp_image = proj_ico.ico2erp_image(tangnet_images_list, erp_image_height)
    image_io.image_save(erp_image, erp_src_image_stitch_filepath)


def test_ico_flow_proj():
    """
    Project the ERP flow to 20 faces flow.
    """
    pass


def test_ico_flow_stitch(input_folder_path, output_path):
    """
    test stitch the icosahedron's 20 face flow.
    """

    pass


if __name__ == "__main__":
    padding_size = 0.3

    tangent_image_size = 481
    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_src_image_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_ico_stitch.png")

    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")
    erp_tar_image_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_ico_stitch.jpg")


    ico_src_image_output_dir = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_ico/")
    ico_tar_image_output_dir = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_ico/")
    if not os.path.exists(ico_src_image_output_dir):
        os.mkdir(ico_src_image_output_dir)
    if not os.path.exists(ico_tar_image_output_dir):
        os.mkdir(ico_tar_image_output_dir)

    tangent_image_filename_expression = "ico_rgb_src_{}.png"

    # 1) test padding size
    test_ico_parameters(padding_size)

    # 2) test the image project and stitch
    # test_ico_image_proj(erp_src_image_filepath, tangent_image_filename_expression, ico_src_image_output_dir, tangent_image_size, padding_size)
    # test_ico_image_stitch(ico_src_image_output_dir, tangent_image_filename_expression, erp_src_image_stitch_filepath)

    # 3) test optical flow project and stitch
    # erp_flow_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    # tangent_flow_filename_expression = "ico_flow_src_{}.jpg"

    # test_ico_flow_proj()
    # test_ico_flow_stitch()
