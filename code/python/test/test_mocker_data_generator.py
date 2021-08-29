import configuration

import mocker_data_generator as MDG
from utility import image_io

def test_get_erp_image():
    """[summary]
    """
    image_data = MDG.get_erp_image()
    image_io.image_show(image_data)


if __name__ == "__main__":
    test_get_erp_image()