import configuration as config

import os

from utility import image_io
from utility import flow_io
from utility import flow_vis

def vis_of_folder(data_dir):
    counter = 0
    for filename in os.listdir(data_dir):
        counter = counter + 1
        if counter % 10 == 0:
            print(f"{counter} : {filename}")

        if filename.endswith(".flo"):  # and filename == "0002_R_motionvector_forward.flo":
            of_data = flow_io.read_flow_flo(data_dir + filename)
            of_data_vis = flow_vis.flow_to_color(of_data,min_ratio=0.2, max_ratio=0.8)#,  min_ratio=0.3, max_ratio=0.97)
            # of_data_vis = flow_vis.flow_value_to_color(of_data, min_ratio=0.2, max_ratio=0.8)
            image_io.image_save(of_data_vis, data_dir + filename + ".jpg")
            # print("visual optical flow {}".format(filename))
            # of_data_vis_uv = flow_vis.flow_max_min_visual(of_data, None)#"D:/1.jpg")


if __name__ == "__main__":
    vis_of_folder("D:/workdata/omniphoto_bmvc_2021/BathAbbey2/result/pwcnet/")
    # of_data = flow_io.readFlowFile("D:/workdata/omniphoto_bmvc_2021/BathAbbey2/result/pwcnet/")
    # image_io.image_show(of_data[:,:,0])
    # image_io.image_show(of_data[:,:,1])
    # of_data_vis = flow_vis.flow_to_color(of_data, [-3, 3])
    # image_io.image_show(of_data_vis)


