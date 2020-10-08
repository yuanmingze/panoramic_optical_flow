import numpy as np

from . import image_io
from . import flow_vis
from . import flow_io


def compute_occlusion():
    """
    
    """
    pass


def convert_warp_around(flow_original):
    """
    Process the optical flow warp around.

    :param flow: the flow without warp around
    :return: corrected flow
    """
    image_height = np.shape(flow_original)[0]
    image_width = np.shape(flow_original)[1]

    flow_u = flow_original[:,:,0]
    index_u = flow_u > (image_width / 2.0)
    flow_u[index_u] = flow_u[index_u] - image_width
    index_u = flow_u < -(image_width / 2.0)
    flow_u[index_u] = flow_u[index_u] + image_width

    flow_v = flow_original[:,:,1]
    index_v = flow_v > (image_height / 2.0)
    flow_v[index_v] = flow_v[index_v] - image_height
    index_v = flow_v < -(image_height / 2.0)
    flow_v[index_v] = flow_v[index_v] + image_height

    return np.stack((flow_u, flow_v), axis =2)


if __name__ == "__main__":
    """
    Test
    """
    of_gt = flow_io.readFlowFile("../../data/replica_360/hotel_0/0001_opticalflow_forward.flo")
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)

    of_gt = convert_warp_around(of_gt)
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)
