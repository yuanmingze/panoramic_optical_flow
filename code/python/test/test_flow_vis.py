import configuration as config

from utility import image_io
from utility import flow_io
from utility import flow_vis

if __name__ == "__main__":
    of_data = flow_io.readFlowFile("/home/mingze/Downloads/0001.flo")
    image_io.image_show(of_data[:,:,0])
    image_io.image_show(of_data[:,:,1])
    of_data_vis = flow_vis.flow_to_color(of_data, [-3, 3])
    image_io.image_show(of_data_vis)


