import os

import configuration as config

from utility import gnomonic_projection
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import projection_icosahedron as proj_ico

def test_ico_image_stitch():
    """
    test stitch the icosahedron's image face.
    """

    pass


def test_ico_flow_stitch():
    """
    test stitch the icosahedron's 20 face flow.
    """

    pass


def test_ico_image_proj(erp_image_filepath, ico_image_output):
    """
    Project the ERP image to 20 faces flow.
    """
    erp_image = image_io.image_read(erp_image_filepath)
    face_images_src = proj_ico.erp2ico_image_gnomonic(erp_image)
    for index in range(0, len(face_images_src)):
        cubemap_images_name = ico_image_output + "ico_rgb_src_{}.jpg".format(index)
        image_io.image_save(face_images_src[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


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
    
    if not os.path.exists(ico_image_output):
        os.mkdir(ico_image_output)

    test_ico_image_proj(erp_image_filepath, ico_image_output)
    # test_ico_flow_proj()


    # test_ico_image_stitch()
    # test_ico_flow_stitch()