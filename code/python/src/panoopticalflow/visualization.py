
import pathlib

from . import flow_io
from . import flow_vis
from . import image_io
from . import depth_io

from .logger import Logger
log = Logger(__name__)
log.logger.propagate = False

def vis_dir(root_dir, recursive_enable=True):
    """ Recursively visualize the data.

    Visual data type:
    *.pfm to *.jpg file,
    *.flo to *.jpg file,

    :param root_dir: The root dir of data.
    :type root_dir: str
    :param recursice: recursive , defaults to True
    :type recursice: bool, optional
    """
    dir_path = pathlib.Path(root_dir)
    if not dir_path.exists():
        log.warn("Directory {} do not exist".format(root_dir))
        return

    for file_path in dir_path.iterdir():
        if file_path.is_dir():
            vis_dir(file_path, recursive_enable)
        else:
            # visualize optical flow
            if file_path.suffix == ".floss" or file_path.suffix == ".flo":
                log.info("visualize file: {}".format(file_path))
                if file_path.suffix == ".floss":
                    of_data = flow_io.read_flow_floss(str(file_path))
                elif file_path.suffix == ".flo":
                    of_data = flow_io.read_flow_flo(str(file_path))

                # TODO judge and visual 360 flow
                of_data_color = flow_vis.flow_to_color(of_data)
                flow_visual_file_path = str(file_path) + ".jpg"
                image_io.image_save(of_data_color, flow_visual_file_path)

            elif file_path.suffix == ".pfm" or file_path.suffix == ".dpt":
                log.info("visualize file: {}".format(file_path))
                # visualize depth map
                if file_path.suffix == ".pfm":
                    depth_data = depth_io.read_pfm(str(file_path))

                elif file_path.suffix == ".dpt":
                    depth_data = depth_io.read_dpt(str(file_path))

                output_path = str(file_path) + ".jpg"
                depth_io.depth_visual_save(depth_data, output_path, min_ratio=0.05, max_ratio=0.95, visual_colormap="jet")
