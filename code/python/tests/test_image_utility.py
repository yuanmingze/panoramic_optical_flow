import configuration as config

from utility import image_utility
from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def test_image_file_resize(data_root):
    image_filepath_input = data_root + "replica_360/apartment_0/0001_rgb.jpg"
    image_filepath_output = data_root + "replica_360/apartment_0/0001_rgb_resized.jpg"
    image_resize_ratio = 0.4
    image_utility.image_file_resize(image_filepath_input, image_filepath_output, image_resize_ratio)


if __name__ == "__main__":
    test_list = [0]
    data_root = config.TEST_data_root_dir
    if 0 in test_list:
        test_image_file_resize(data_root)
