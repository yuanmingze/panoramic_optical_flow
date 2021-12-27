import configuration as config

import draft_figure


def test_bunny_animation():
    input_image_filepath = config.TEST_data_root_dir + "bunny_src_image.png"
    output_image_filepath = config.TEST_data_root_dir + "bunny_animation/output.gif"
    draft_figure.bunny_animation(input_image_filepath, output_image_filepath)

if __name__ == "__main__":
    test_bunny_animation()
