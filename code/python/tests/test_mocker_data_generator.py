import configuration

from utility import mocker_data_generator as MDG
from utility import image_io

def test_get_erp_image():
    """[summary]
    """
    image_data = MDG.image_strip()
    image_io.image_show(image_data)

def test_get_demo_erp_image():
    image_data = MDG.image_square()
    image_io.image_show(image_data)


if __name__ == "__main__":
    # test_get_erp_image()
    test_get_demo_erp_image()