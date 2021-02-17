import configuration as config
import flow_post_proc

from utility import image_io
from utility import flow_io
from utility import flow_vis

if __name__ == "__main__":
    """
    Test
    """
    of_gt = flow_io.readFlowFile("../../data/replica_360/hotel_0/0001_opticalflow_forward.flo")
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)

    of_gt =  flow_post_proc.convert_warp_around(of_gt)
    of_gt_vis = flow_vis.flow_to_color(of_gt)
    image_io.image_show(of_gt_vis)
