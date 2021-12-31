import configuration as config

import os

from panoopticalflow import depth_io
from panoopticalflow.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def test_vis_dpt_folder(data_dir):
    counter = 0
    for filename in os.listdir(data_dir):
        counter = counter + 1
        if counter % 10 == 0:
            print(f"{counter} : {filename}")

        if filename.endswith(".dpt"):  # and filename == "0002_R_motionvector_forward.flo":
            depth_data = depth_io.read_dpt(data_dir + filename)
            output_path = data_dir + filename + ".jpg"
            depth_io.depth_visual_save(depth_data, output_path, min_ratio=0.05, max_ratio=0.95, visual_colormap="jet")


if __name__ == "__main__":
    # data_dir = config.TEST_data_root_dir + "replica_360/office_0_line_cubemap_stitch_debug/cubemap_dpt/"
    # data_dir = "D:/workdata/opticalflow_data/replica_cubemap/office_0_line/"
    data_dir = config.TEST_data_root_dir + "replica_360/office_0_line_cubemap_stitch_debug/"
    test_vis_dpt_folder(data_dir)
