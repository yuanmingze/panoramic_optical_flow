import os
import pathlib

from utility import flow_io, flow_vis, image_io, depth_io


def of_vis(of_data_dir):
    """
    """
    of_data_path = pathlib.Path(of_data_dir)
    for file_path in of_data_path.iterdir():
        if not file_path.suffix == ".floss":
            continue

        of_data = flow_io.readFlowFloss(str(file_path))
        of_data_color = flow_vis.flow_to_color(of_data)

        flow_visual_file_path = str(file_path) + ".jpg"
        image_io.image_save(of_data_color, flow_visual_file_path)


def depth_map_format_transfrom():
    """
    """
    root_dir = "D:/workdata/casual_stereo_vr_2020_test/central_viewpoint_rgbd/library/"
    depth_dpt_path = root_dir + "0000_depth.dpt"
    depth_png_path = root_dir + "0000_depth.png"
    depth_visual_path = root_dir + "0000_depth.jpg"

    depth_data = depth_io.read_dpt(depth_dpt_path)

    depth_visual = depth_io.depth_visual(depth_data)
    image_io.image_save(depth_visual, depth_visual_path)

    depth_io.write_png(depth_data, depth_png_path)


if __name__ == "__main__":
    #root_folder = "D:/workdata/casual_stereo_vr_2020_test/boatshed_colmap_00_below_omni/Cache/29-2k-2k-DIS/"
    #root_folder = "D:/workdata/casual_stereo_vr_2020_test/bm_colmap_result_00_upper_omni/Cache/25-2k-2k-DIS/"
    # root_folder = "D:/workdata/casual_stereo_vr_2020_test/gasworks_colmap_00_below_omni/Cache/29-2k-2k-DIS/"
    # root_folder = "D:/workdata/casual_stereo_vr_2020_test/boatshed_colmap_00_below_omni_pano/Cache/29-2k-2k-DIS/"
    # of_vis(root_folder)
    depth_map_format_transfrom()