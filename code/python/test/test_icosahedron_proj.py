import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import configuration as config

from utility import image_io
from utility import projection_icosahedron as proj_ico
from utility import flow_io
from utility import flow_vis
from utility import flow_warp
from utility import flow_estimate

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def compute_ico_faces_DIS(src_folder_path, src_file_expression,
                          dest_folder_path, dest_file_expression,
                          flow_dis_output_path, flow_dis_expression,
                          face_number=20):
    """Estimate the flow for each face image. output to cubemap source images' folder.
    The post-fix is dis.
    """
    src_image_list = []
    tar_image_list = []

    # 1) load images
    for index in range(0, face_number):
        face_src_image_name = src_file_expression.format(index)
        src_image_list.append(image_io.image_read(os.path.join(src_folder_path, face_src_image_name)))

        face_tar_image_name = dest_file_expression.format(index)
        tar_image_list.append(image_io.image_read(os.path.join(dest_folder_path, face_tar_image_name)))

    # 2) estimate them DIS optical flow
    for index in range(0, face_number):
        # import ipdb; ipdb.set_trace()
        face_flow = flow_estimate.DIS(src_image_list[index], tar_image_list[index])

        face_flow_path = os.path.join(flow_dis_output_path, flow_dis_expression.format(index))
        flow_io.flow_write(face_flow, face_flow_path)
        face_flow_vis = flow_vis.flow_to_color(face_flow)
        image_io.image_save(face_flow_vis, face_flow_path + ".jpg")
        log.debug("output the DIS flow to {}".format(face_flow_path))

        # warp image
        print("output warped result: "+src_file_expression.format(index))
        src_image_path = src_folder_path + src_file_expression.format(index)
        image_src = image_io.image_read(src_image_path)
        image_src_warpped = flow_warp.warp_forward(image_src, face_flow)
        image_src_warpped_path = src_folder_path + src_file_expression.format(index) + "_warp_dis.png"
        image_io.image_save(image_src_warpped, image_src_warpped_path)


def test_ico_parameters(padding_size):
    """
    Check the icosahedron's paramters.
    """
    def plot_points(ax, point_list, line_style='ro-'):
        triangle_points_tangent_origin_list_x = []
        triangle_points_tangent_origin_list_y = []
        for index in range(1 + len(point_list)):
            index_new = index % len(point_list)
            triangle_points_tangent_origin_list_x.append(point_list[index_new][0])
            triangle_points_tangent_origin_list_y.append(point_list[index_new][1])
        print("index:{}, tangent_point:{}, triangle_points_sph:{}".format(triangle_index, ico_parameter_origin["triangle_points_tangent"], ico_parameter_origin["triangle_points_sph"]))
        # ax.scatter(triangle_points_tangent_origin_list_x, triangle_points_tangent_origin_list_y)
        ax.plot(triangle_points_tangent_origin_list_x, triangle_points_tangent_origin_list_y, line_style, linewidth=3)

    print("padding size {}".format(padding_size))
    fig, axs = plt.subplots(4, 5)
    for triangle_index in range(0, 20):
        plt_title = "The tangent triangle {}".format(triangle_index)
        ax = axs[int(triangle_index / 5)][int(triangle_index % 5)]
        ax.set_title(plt_title)

        #  plot the pading points and original points
        # 0) plot the origin points
        # plot the parameter
        ico_parameter_origin = proj_ico.get_icosahedron_parameters(triangle_index, 0.0)
        # tangent_point_origin = ico_parameter_origin["tangent_point"]
        triangle_points_tangent_origin = ico_parameter_origin["triangle_points_tangent"]
        # triangle_points_sph_origin =     ico_parameter_origin["triangle_points_sph"]
        # availied_ERP_area_origin =       ico_parameter_origin["availied_ERP_area"]
        plot_points(ax, triangle_points_tangent_origin, 'ro-')

        # 1) plot the padding points
        ico_parameter = proj_ico.get_icosahedron_parameters(triangle_index, padding_size)
        # tangent_point = ico_parameter["tangent_point"]
        triangle_points_tangent = ico_parameter["triangle_points_tangent"]
        # triangle_points_sph = ico_parameter["triangle_points_sph"]
        # availied_ERP_area = ico_parameter["availied_ERP_area"]
        plot_points(ax, triangle_points_tangent, 'go-')

    plt.show()


def test_ico_image_proj(erp_image_filepath, ico_images_expression, ico_image_output, tangent_image_size, padding_size):
    """
    Project the ERP image to 20 faces flow.
    """
    # test
    erp_image = image_io.image_read(erp_image_filepath)

    tangent_image_list = proj_ico.erp2ico_image(erp_image, tangent_image_size, padding_size, full_face_image=True)
    for index in range(0, len(tangent_image_list)):
        cubemap_images_name = ico_image_output + ico_images_expression.format(index)
        image_io.image_save(tangent_image_list[index], cubemap_images_name)
        # image_io.image_show(face_images[0])


def test_ico_image_stitch(input_folder_path, file_expression, erp_src_image_stitch_filepath, erp_image_height, padding_size):
    """
    test stitch the icosahedron's image face.
    """
    # load the 20 tangnet images
    tangnet_images_list = []
    for index in range(0, 20):
        tangnet_images_name = input_folder_path + file_expression.format(index)
        tangnet_images_list.append(image_io.image_read(tangnet_images_name))

    # stitch image
    erp_image = proj_ico.ico2erp_image(tangnet_images_list, erp_image_height, padding_size, "mean")
    image_io.image_save(erp_image, erp_src_image_stitch_filepath)


def test_ico_flow_proj(erp_flow_filepath, ico_src_image_output_dir, tangent_flow_filename_expression, tangent_image_filename_expression, tangent_image_size, padding_size):
    """
    Project the ERP flow to 20 faces flow.
    Test stitch 20 face optical flow to single ERP flow.
    """
    if not os.path.exists(ico_src_image_output_dir):
        os.mkdir(ico_src_image_output_dir)

    # 1) ERP flow to cubemap flow
    erp_flow = flow_io.read_flow_flo(erp_flow_filepath)
    face_flows = proj_ico.erp2ico_flow(erp_flow, tangent_image_size, padding_size)
    for index in range(0, len(face_flows)):
        ico_flow_name = ico_src_image_output_dir + tangent_flow_filename_expression.format(index)
        flow_io.write_flow_flo(face_flows[index], ico_flow_name)

        ico_flow_vis_name = ico_src_image_output_dir + (tangent_flow_filename_expression + "_vis.jpg").format(index)
        face_flow_vis = flow_vis.flow_to_color(face_flows[index], [-300, 300])
        # image_io.image_show(face_flow_vis)
        image_io.image_save(face_flow_vis, ico_flow_vis_name)

        # warp image
        print("output warped result: "+tangent_image_filename_expression.format(index))
        src_image_path = ico_src_image_output_dir + tangent_image_filename_expression.format(index)
        image_src = image_io.image_read(src_image_path)
        image_src_warpped = flow_warp.warp_forward(image_src, face_flows[index])
        image_src_warpped_path = ico_src_image_output_dir + tangent_image_filename_expression.format(index) + "_warp.png"
        image_io.image_save(image_src_warpped, image_src_warpped_path)


def test_ico_flow_stitch(ico_src_image_output_dir, tangent_flow_filename_expression, erp_src_flow_stitch_filepath,
                         erp_image_height, padding_size,
                         erp_src_image_stitch_filepath,
                         src_erp_image_path, tar_erp_image_path):
    """
    test stitch the icosahedron's 20 face flow.
    """
    # 1) load the face's flow file to memory
    face_flows = []
    for index in range(0, 20):
        cubemap_flow_path = ico_src_image_output_dir + tangent_flow_filename_expression.format(index)
        face_flows.append(flow_io.read_flow_flo(cubemap_flow_path))

    # 2) stitch the faces flow to ERP flow
    image_erp_src = image_io.image_read(src_erp_image_path)
    image_erp_tar = image_io.image_read(tar_erp_image_path)

    erp_flow_stitch = proj_ico.ico2erp_flow(face_flows, erp_image_height, padding_size, image_erp_src=image_erp_src, image_erp_tar=image_erp_tar)
    flow_io.write_flow_flo(erp_flow_stitch, erp_src_flow_stitch_filepath)

    face_flow_vis = flow_vis.flow_to_color(erp_flow_stitch, [-40, 40])
    # image_io.image_show(face_flow_vis)
    log.info("output flow to: {}".format(erp_src_flow_stitch_filepath + "_vis.jpg"))
    image_io.image_save(face_flow_vis, erp_src_flow_stitch_filepath + "_vis.jpg")

    # 3) test the optical flow with warp
    print("output warped result: " + erp_src_image_stitch_filepath)
    image_src = image_io.image_read(erp_src_image_stitch_filepath)
    if image_src.shape[0:2] != erp_flow_stitch.shape[0:2]:
        image_src = Image.fromarray(obj=image_src, mode='RGB').resize((erp_flow_stitch.shape[1], erp_flow_stitch.shape[0]))
        image_src = np.array(image_src)

    image_src_warpped = flow_warp.warp_forward(image_src, erp_flow_stitch)
    # TODO the result look wired check !
    image_src_warpped_path = erp_src_image_stitch_filepath + "_ico_stitch_warp.png"
    image_io.image_save(image_src_warpped, image_src_warpped_path)


if __name__ == "__main__":
    padding_size = 0.1
    ico_face_number = 20

    tangent_image_size = 480
    erp_image_height = 960
    erp_image_width = erp_image_height * 2

    erp_src_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb.jpg")
    erp_src_image_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_ico_stitch.png")

    erp_tar_image_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb.jpg")
    erp_tar_image_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_ico_stitch.jpg")

    ico_src_image_output_dir = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_rgb_ico/")
    ico_tar_image_output_dir = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0002_rgb_ico/")
    if not os.path.exists(ico_src_image_output_dir):
        os.mkdir(ico_src_image_output_dir)
    if not os.path.exists(ico_tar_image_output_dir):
        os.mkdir(ico_tar_image_output_dir)

    tangent_image_filename_expression = "ico_rgb_src_{}.png"
    tangent_padding_image_filename_expression = "ico_rgb_src_padding_{}.png"

    # 1) test padding size
    test_ico_parameters(padding_size)

    # 2) test the image project and stitch
    test_ico_image_proj(erp_src_image_filepath, tangent_image_filename_expression, ico_src_image_output_dir, tangent_image_size, padding_size)
    test_ico_image_stitch(ico_src_image_output_dir, tangent_image_filename_expression, erp_src_image_stitch_filepath, erp_image_height, padding_size)
    test_ico_image_proj(erp_tar_image_filepath, tangent_image_filename_expression, ico_tar_image_output_dir, tangent_image_size, padding_size)

    # 3) test the image projection and stitch with padding
    test_ico_image_proj(erp_src_image_filepath, tangent_padding_image_filename_expression, ico_src_image_output_dir, tangent_image_size, padding_size)
    test_ico_image_stitch(ico_src_image_output_dir, tangent_padding_image_filename_expression, erp_src_image_stitch_filepath, erp_image_height, padding_size)

    # 4) test optical flow project and stitch
    erp_flow_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward.flo")
    erp_flow_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_forward_stitch.flo")
    tangent_flow_filename_expression = "ico_flow_src_{}.flo"

    test_ico_flow_proj(erp_flow_filepath, ico_src_image_output_dir, tangent_flow_filename_expression, tangent_image_filename_expression, tangent_image_size, padding_size)
    test_ico_flow_stitch(ico_src_image_output_dir, tangent_flow_filename_expression, erp_flow_stitch_filepath, erp_image_height, padding_size, erp_src_image_filepath, erp_src_image_filepath, erp_tar_image_filepath)

    # 5) compute the each face's optical flow with DIS
    tangent_DIS_flow_filename_expression = "ico_flow_dis_src_{}.flo"
    compute_ico_faces_DIS(ico_src_image_output_dir, tangent_image_filename_expression,
                        ico_tar_image_output_dir, tangent_image_filename_expression,
                        ico_src_image_output_dir, tangent_DIS_flow_filename_expression)

    # 6) stitch all face DIS optical flow to ERP flow
    erp_flow_dis_stitch_filepath = os.path.join(config.TEST_data_root_dir, "replica_360/apartment_0/0001_opticalflow_dis_forward_stitch.flo")
    test_ico_flow_stitch(ico_src_image_output_dir, tangent_DIS_flow_filename_expression,
                erp_flow_dis_stitch_filepath, erp_image_height, padding_size, erp_src_image_filepath,
                erp_src_image_filepath, erp_tar_image_filepath)
