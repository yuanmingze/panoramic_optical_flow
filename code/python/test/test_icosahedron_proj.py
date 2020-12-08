import os

import configuration as config

from utility import image_io
from utility import projection_icosahedron as proj_ico

def test_ico_parameters():
    """
    Check the icosahedron's paramters.
    """
    
    


def test_ico_image_stitch(input_folder_path, file_expression, output_folder_path):
    """
    test stitch the icosahedron's image face.
    """
    erp_image_height = 960

    # load the 20 tangnet images
    tangnet_images_list = []
    for index in range(0, 20):
        tangnet_images_name = input_folder_path + file_expression.format(index)
        tangnet_images_list.append(image_io.image_read(tangnet_images_name))

    # stitch image
    erp_image = proj_ico.ico2erp_image_gnomonic(tangnet_images_list, erp_image_height)
    image_io.image_save( erp_image, output_folder_path + "erp_image_stitched.png")


def test_ico_flow_stitch(input_folder_path, output_path):
    """
    test stitch the icosahedron's 20 face flow.
    """



    pass


def test_ico_image_proj(erp_image_filepath, ico_images_expression, ico_image_output):
    """
    Project the ERP image to 20 faces flow.
    """
    erp_image = image_io.image_read(erp_image_filepath)
    tangent_image_size = 480
    tangent_image_list = proj_ico.erp2ico_image_gnomonic(erp_image, tangent_image_size)
    for index in range(0, len(tangent_image_list)):
        cubemap_images_name = ico_image_output + ico_images_expression.format(index)

        image_io.image_save(tangent_image_list[index], cubemap_images_name)
        # image_io.image_show(face_images[0])

    # for index in range(0,20):
    #     ico_parameter = proj_ico.get_icosahedron_parameters(index)
    #     print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(index, ico_parameter["triangle_points_tangent"], ico_parameter["triangle_points_sph"]))



def test_ico_flow_proj():
    """
    Project the ERP flow to 20 faces flow.
    """
    pass


if __name__ == "__main__":
    erp_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_flow_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")

    ico_image_output = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_ico/")
    ico_flow_output = ico_image_output
    
    tangent_image_filename_expression = "new_ico_rgb_src_{}.png"

    if not os.path.exists(ico_image_output):
        os.mkdir(ico_image_output)

    # test_ico_image_proj(erp_image_filepath, tangent_image_filename_expression, ico_image_output)
    test_ico_image_stitch(ico_image_output, tangent_image_filename_expression, ico_image_output)

    # test_ico_flow_proj()
    # test_ico_flow_stitch()