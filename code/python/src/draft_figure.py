import image_io
import spherical_coordinates

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def bunny_animation(input_image_filepath, output_image_filepath):
    """ Make a bunny animation gif figure.

    :param input_image_filepath: The input 
    :type input_image_filepath: str
    :param output_image_filepath: The gif output file path.
    :type output_image_filepath: str
    """
    src_image_data = image_io.image_read(input_image_filepath)
    if src_image_data.shape[2] == 4:
        src_image_data = src_image_data[:, :, 0:3]

    frame_rate = 25
    time_duration = 3
    # rotation_degree = 360.0 / (frame_rate * time_duration)

    angle_start = 0
    angle_end = 120
    rotation_degree = (angle_end - angle_start) / (frame_rate * time_duration)

    # 0) create image sequence
    image_seq_list = []
    image_intermedia = src_image_data
    image_seq_list.append(image_intermedia)

    for idx in range(frame_rate * time_duration):
        image_intermedia = spherical_coordinates.rotation_erp_horizontal_fast(image_intermedia, rotation_degree)
        image_seq_list.append(image_intermedia)

        # # for debug output each image
        # if True:
        #     output_image_filepath_temp = output_image_filepath + "{}.jpg".format(idx)
        #     log.info("output image {}".format(output_image_filepath_temp))
        #     image_io.image_save(image_intermedia, output_image_filepath_temp)

    # 1) create fig animation image
    image_io.image_seq2gif(image_seq_list, output_image_filepath)
