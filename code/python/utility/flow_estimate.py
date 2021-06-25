import cv2
import numpy as np
import flow_postproc
import flow_vis
import flow_io

import projection
import image_io
import projection_cubemap as proj_cm
import projection_icosahedron as proj_ico

from logger import Logger
import spherical_coordinates

log = Logger(__name__)
log.logger.propagate = False


def of_methdod_DIS(image_src_original, image_tar_original):
    """Compute the DIS flow.

    :param image_src_original: The optical flow source image.
    :type image_src_original: numpy
    :param image_tar_original: The optical flow target image.
    :type image_tar_original: numpy
    :return: the optical flow.
    :rtype: numpy
    """
    image_src = image_src_original
    image_tar = image_tar_original

    # the image including alpha channel
    if image_src.shape[2] == 4:
        image_src = image_src[:, :, :3]
    if image_tar.shape[2] == 4:
        image_tar = image_tar[:, :, :3]

    # convert image dtype
    if image_src.dtype != np.uint8:
        image_src = image_src.astype(np.uint8)
    if image_tar.dtype != np.uint8:
        image_tar = image_tar.astype(np.uint8)

    # RGB to gray
    if image_src.shape[2] == 3:
        image_src_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_src_gray = image_src

    if image_tar.shape[2] == 3:
        image_tar_gray = cv2.cvtColor(image_tar, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_tar_gray = image_tar

    inst = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(True)

    return inst.calc(image_src_gray, image_tar_gray, None)


def debug_save_of(of_data, output_filepath):
    """Visualize optical flow both warp-around and un-warp-around."""
    # flow_vis.flow_value_to_color(of_data)
    min_ratio = 0.2
    max_ratio = 0.8
    of_data_visual = flow_vis.flow_to_color(of_data, min_ratio=min_ratio, max_ratio=max_ratio)
    image_io.image_save(of_data_visual, output_filepath + "_flow_wraparound.jpg")
    # of_data_warparound = flow_postproc.erp_of_unwraparound(of_data)
    of_data_warparound = flow_postproc.erp_of_wraparound(of_data)
    of_data_warparound_visual = flow_vis.flow_to_color(of_data_warparound, min_ratio=min_ratio, max_ratio=max_ratio)
    image_io.image_save(of_data_warparound_visual, output_filepath + "_flow_unwraparound.jpg")


def pano_of_0(src_erp_image, tar_erp_image, optical_flow_method=None, debug_output_dir=None, face_blending_method="straightforward"):
    """Compute the optical flow with multi-step and icosahedron projection.

    The multi-steps method, just transfer the ERP image rotation to next step.
    The process use warp-around (overflowed) optical flow.

    :param src_erp_image: the source ERP image data.
    :type src_erp_image: numpy
    :param tar_erp_image: the target ERP image data.
    :type tar_erp_image: numpy
    :param optical_flow_method: the optical flow estimation function.
    :type: function
    :param debug_output_dir: The folder storing output debug information.
    :type debug_output_dir: str
    :param face_blending_method: the weight to blending faces, default is "straightforward".
    :type face_blending_method: str
    :return: the optical flow from src image to tar image.
    :rtype: numpy
    """
    if optical_flow_method == None:
        optical_flow_method = of_methdod_DIS

    padding_size_cubemap = 0.1
    padding_size_ico = 0.1
    erp_image_height = src_erp_image.shape[0]
    tangent_image_width_ico = 480

    # 0) compute flow in ERP image & wrap image, make it align with the source ERP image.
    log.debug("0) compute ERP image flow")
    optical_flow_erp = optical_flow_method(src_erp_image, tar_erp_image)
    tar_erp_image_rot_erp, erp_rot_theta, erp_rot_phi = projection.image_rotate_flow(tar_erp_image, optical_flow_erp, src_image=False)
    erp_rot_mat = spherical_coordinates.rot_sph2mat(erp_rot_theta, erp_rot_phi)
    log.debug("ERP optical flow rotation is {}, {}".format(np.degrees(erp_rot_theta), np.degrees(erp_rot_phi)))

    if debug_output_dir is not None:
        debug_save_of(optical_flow_erp, debug_output_dir + "pano_of_0_of_erp")
        image_io.image_save(tar_erp_image_rot_erp, debug_output_dir + "pano_of_0_erp_rot.jpg")

    # 1) compute flow with cubemap projection & warp target image
    log.debug("1) compute cubemap projection image flow")
    # 1-1) erp image to cube map
    cubeface_images_src_list = proj_cm.erp2cubemap_image(src_erp_image, padding_size_cubemap)
    cubeface_images_tar_list = proj_cm.erp2cubemap_image(tar_erp_image_rot_erp, padding_size_cubemap)
    cubemap_face_of_list = []
    for index in range(0, len(cubeface_images_src_list)):
        optical_flow_cubemap = optical_flow_method(cubeface_images_src_list[index], cubeface_images_tar_list[index])
        cubemap_face_of_list.append(optical_flow_cubemap)
    optical_flow_cubemap = proj_cm.cubemap2erp_flow(cubemap_face_of_list, erp_image_height, padding_size_cubemap, src_erp_image, tar_erp_image, wrap_around=True)
    # 1-2) warp target image
    tar_erp_image_rot_cubemap, cubemap_rot_theta, cubemap_rot_phi = projection.image_rotate_flow(tar_erp_image_rot_erp, optical_flow_cubemap, src_image=False)
    cubemap_rot_mat = spherical_coordinates.rot_sph2mat(cubemap_rot_theta, cubemap_rot_phi)
    log.debug("Cubemap optical flow rotation is {}, {}".format(np.degrees(cubemap_rot_theta), np.degrees(cubemap_rot_phi)))

    if debug_output_dir is not None:
        debug_save_of(optical_flow_cubemap, debug_output_dir + "pano_of_0_of_cubemap")
        image_io.image_save(tar_erp_image_rot_cubemap, debug_output_dir + "pano_of_0_cubemap_rot.jpg")
        # for index in range(len(cubemap_face_of_list)):
        #     flow_io.flow_write(cubemap_face_of_list[index], debug_output_dir + "cubemap_flow_padding_{}.flo".format(index))
        #     of_data_visual = flow_vis.flow_to_color(cubemap_face_of_list[index])
        #     image_io.image_save(of_data_visual,  debug_output_dir + "cubemap_flow_padding_{}.jpg".format(index))
        # for index in range(len(cubeface_images_src_list)):
        #     image_io.image_save(cubeface_images_src_list[index],  debug_output_dir + "cubemap_src_subimage_{}.jpg".format(index))
        # for index in range(len(cubeface_images_tar_list)):
        #     image_io.image_save(cubeface_images_tar_list[index],  debug_output_dir + "cubemap_tar_subimage_{}.jpg".format(index))

    # 2) compute flow with icosahedron projection & warp image
    log.debug("2) compute icosahedron projection image flow")
    # 2-1) erp image to cube map
    icoface_images_src_list = proj_ico.erp2ico_image(src_erp_image, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    icoface_images_tar_list = proj_ico.erp2ico_image(tar_erp_image_rot_cubemap, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    ico_face_of_list = []
    for index in range(0, len(icoface_images_src_list)):
        optical_flow_ico = optical_flow_method(icoface_images_src_list[index], icoface_images_tar_list[index])
        ico_face_of_list.append(optical_flow_ico)
    optical_flow_ico = proj_ico.ico2erp_flow(ico_face_of_list, erp_image_height, padding_size_ico, src_erp_image, tar_erp_image, wrap_around=True, face_blending_method=face_blending_method)

    # optical_flow_ico = flow_postproc.erp_of_wraparound(optical_flow_ico)
    # 2-2) warp target image
    #tar_erp_image_rot_ico, ico_rot_theta, ico_rot_phi = projection.image_rotate_flow(tar_erp_image_rot_cubemap, optical_flow_ico)

    if debug_output_dir is not None:
        debug_save_of(optical_flow_ico, debug_output_dir + "pano_of_0_of_ico")
        # for index in range(len(ico_face_of_list)):
        #     of_data_visual = flow_vis.flow_to_color(ico_face_of_list[index])
        #     image_io.image_save(of_data_visual,  debug_output_dir + "ico_flow_padding_{}.jpg".format(index))
        # for index in range(len(icoface_images_src_list)):
        #     image_io.image_save(icoface_images_src_list[index],  debug_output_dir + "ico_src_subimage_{}.png".format(index))
        # for index in range(len(icoface_images_tar_list)):
        #     image_io.image_save(icoface_images_tar_list[index],  debug_output_dir + "ico_tar_subimage_{}.png".format(index))

    # 3) accumulate all-steps optical flow
    of_ico2cub = projection.flow_rotate_endpoint(optical_flow_ico, cubemap_rot_mat.T)
    of_accu = projection.flow_rotate_endpoint(of_ico2cub, erp_rot_mat.T)

    if debug_output_dir is not None:
        debug_save_of(of_ico2cub, debug_output_dir + "pano_of_0_ico_of_ico2cub")
        debug_save_of(of_accu, debug_output_dir + "pano_of_0_ico_of_accu")
        
    return of_accu


def pano_of_0_wo_erp(src_erp_image, tar_erp_image, optical_flow_method=None, debug_output_dir=None, face_blending_method="straightforward"):
    """Compute the optical flow with multi-step and icosahedron projection.

    The multi-steps method, just transfer the ERP image rotation to next step.
    The process use warp-around (overflowed) optical flow.

    :param src_erp_image: the source ERP image data.
    :type src_erp_image: numpy
    :param tar_erp_image: the target ERP image data.
    :type tar_erp_image: numpy
    :param optical_flow_method: the optical flow estimation function.
    :type: function
    :param debug_output_dir: The folder storing output debug information.
    :type debug_output_dir: str
    :param face_blending_method: the weight to blending faces, default is "straightforward".
    :type face_blending_method: str
    :return: the optical flow from src image to tar image.
    :rtype: numpy
    """
    if optical_flow_method == None:
        optical_flow_method = of_methdod_DIS

    padding_size_cubemap = 0.1
    padding_size_ico = 0.1
    erp_image_height = src_erp_image.shape[0]
    tangent_image_width_ico = 480

    # 1) compute flow with cubemap projection & warp target image
    log.debug("1) compute cubemap projection image flow")
    # 1-1) erp image to cube map
    cubeface_images_src_list = proj_cm.erp2cubemap_image(src_erp_image, padding_size_cubemap)
    cubeface_images_tar_list = proj_cm.erp2cubemap_image(tar_erp_image, padding_size_cubemap)
    cubemap_face_of_list = []
    for index in range(0, len(cubeface_images_src_list)):
        optical_flow_cubemap = optical_flow_method(cubeface_images_src_list[index], cubeface_images_tar_list[index])
        cubemap_face_of_list.append(optical_flow_cubemap)
    optical_flow_cubemap = proj_cm.cubemap2erp_flow(cubemap_face_of_list, erp_image_height, padding_size_cubemap, src_erp_image, tar_erp_image, wrap_around=True)
    # 1-2) warp target image
    tar_erp_image_rot_cubemap, cubemap_rot_theta, cubemap_rot_phi = projection.image_rotate_flow(tar_erp_image, optical_flow_cubemap, src_image=False)
    cubemap_rot_mat = spherical_coordinates.rot_sph2mat(cubemap_rot_theta, cubemap_rot_phi)
    log.debug("Cubemap optical flow rotation is {}, {}".format(np.degrees(cubemap_rot_theta), np.degrees(cubemap_rot_phi)))

    # 2) compute flow with icosahedron projection & warp image
    log.debug("2) compute icosahedron projection image flow")
    # 2-1) erp image to cube map
    icoface_images_src_list = proj_ico.erp2ico_image(src_erp_image, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    icoface_images_tar_list = proj_ico.erp2ico_image(tar_erp_image_rot_cubemap, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    ico_face_of_list = []
    for index in range(0, len(icoface_images_src_list)):
        optical_flow_ico = optical_flow_method(icoface_images_src_list[index], icoface_images_tar_list[index])
        ico_face_of_list.append(optical_flow_ico)
    optical_flow_ico = proj_ico.ico2erp_flow(ico_face_of_list, erp_image_height, padding_size_ico, src_erp_image, tar_erp_image, wrap_around=True, face_blending_method=face_blending_method)

    # 3) accumulate all-steps optical flow
    of_accu = projection.flow_rotate_endpoint(optical_flow_ico, cubemap_rot_mat.T)

    if debug_output_dir is not None:
        debug_save_of(of_accu, debug_output_dir + "pano_of_0_ico_of_accu")
        
    return of_accu
    
def pano_of_0_wo_cube(src_erp_image, tar_erp_image, optical_flow_method=None, debug_output_dir=None, face_blending_method="straightforward"):
    """Compute the optical flow with multi-step and icosahedron projection.

    The multi-steps method, just transfer the ERP image rotation to next step.
    The process use warp-around (overflowed) optical flow.

    :param src_erp_image: the source ERP image data.
    :type src_erp_image: numpy
    :param tar_erp_image: the target ERP image data.
    :type tar_erp_image: numpy
    :param optical_flow_method: the optical flow estimation function.
    :type: function
    :param debug_output_dir: The folder storing output debug information.
    :type debug_output_dir: str
    :param face_blending_method: the weight to blending faces, default is "straightforward".
    :type face_blending_method: str
    :return: the optical flow from src image to tar image.
    :rtype: numpy
    """
    if optical_flow_method == None:
        optical_flow_method = of_methdod_DIS

    padding_size_ico = 0.1
    erp_image_height = src_erp_image.shape[0]
    tangent_image_width_ico = 480

    # 0) compute flow in ERP image & wrap image, make it align with the source ERP image.
    log.debug("0) compute ERP image flow")
    optical_flow_erp = optical_flow_method(src_erp_image, tar_erp_image)
    tar_erp_image_rot_erp, erp_rot_theta, erp_rot_phi = projection.image_rotate_flow(tar_erp_image, optical_flow_erp, src_image=False)
    erp_rot_mat = spherical_coordinates.rot_sph2mat(erp_rot_theta, erp_rot_phi)
    log.debug("ERP optical flow rotation is {}, {}".format(np.degrees(erp_rot_theta), np.degrees(erp_rot_phi)))

    if debug_output_dir is not None:
        debug_save_of(optical_flow_erp, debug_output_dir + "pano_of_0_of_erp")
        image_io.image_save(tar_erp_image_rot_erp, debug_output_dir + "pano_of_0_erp_rot.jpg")

    # 1) compute flow with icosahedron projection & warp image
    log.debug("2) compute icosahedron projection image flow")
    # 2-1) erp image to cube map
    icoface_images_src_list = proj_ico.erp2ico_image(src_erp_image, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    icoface_images_tar_list = proj_ico.erp2ico_image(tar_erp_image_rot_erp, tangent_image_width_ico, padding_size_ico, full_face_image=True)
    ico_face_of_list = []
    for index in range(0, len(icoface_images_src_list)):
        optical_flow_ico = optical_flow_method(icoface_images_src_list[index], icoface_images_tar_list[index])
        ico_face_of_list.append(optical_flow_ico)
    optical_flow_ico = proj_ico.ico2erp_flow(ico_face_of_list, erp_image_height, padding_size_ico, src_erp_image, tar_erp_image, wrap_around=True, face_blending_method=face_blending_method)

    # 3) accumulate all-steps optical flow
    of_accu = projection.flow_rotate_endpoint(optical_flow_ico, erp_rot_mat.T)
    if debug_output_dir is not None:
        debug_save_of(of_accu, debug_output_dir + "pano_of_0_ico_of_accu")
        
    return of_accu

def pano_of_0_wo_ico(src_erp_image, tar_erp_image, optical_flow_method=None, debug_output_dir=None):
    """Compute the optical flow with multi-step and icosahedron projection.

    The multi-steps method, just transfer the ERP image rotation to next step.
    The process use warp-around (overflowed) optical flow.

    :param src_erp_image: the source ERP image data.
    :type src_erp_image: numpy
    :param tar_erp_image: the target ERP image data.
    :type tar_erp_image: numpy
    :param optical_flow_method: the optical flow estimation function.
    :type: function
    :param debug_output_dir: The folder storing output debug information.
    :type debug_output_dir: str
    :param face_blending_method: the weight to blending faces, default is "straightforward".
    :type face_blending_method: str
    :return: the optical flow from src image to tar image.
    :rtype: numpy
    """
    if optical_flow_method == None:
        optical_flow_method = of_methdod_DIS

    padding_size_cubemap = 0.1
    erp_image_height = src_erp_image.shape[0]

    # 0) compute flow in ERP image & wrap image, make it align with the source ERP image.
    log.debug("0) compute ERP image flow")
    optical_flow_erp = optical_flow_method(src_erp_image, tar_erp_image)
    tar_erp_image_rot_erp, erp_rot_theta, erp_rot_phi = projection.image_rotate_flow(tar_erp_image, optical_flow_erp, src_image=False)
    erp_rot_mat = spherical_coordinates.rot_sph2mat(erp_rot_theta, erp_rot_phi)
    log.debug("ERP optical flow rotation is {}, {}".format(np.degrees(erp_rot_theta), np.degrees(erp_rot_phi)))

    if debug_output_dir is not None:
        debug_save_of(optical_flow_erp, debug_output_dir + "pano_of_0_of_erp")
        image_io.image_save(tar_erp_image_rot_erp, debug_output_dir + "pano_of_0_erp_rot.jpg")

    # 1) compute flow with cubemap projection & warp target image
    log.debug("1) compute cubemap projection image flow")
    # 1-1) erp image to cube map
    cubeface_images_src_list = proj_cm.erp2cubemap_image(src_erp_image, padding_size_cubemap)
    cubeface_images_tar_list = proj_cm.erp2cubemap_image(tar_erp_image_rot_erp, padding_size_cubemap)
    cubemap_face_of_list = []
    for index in range(0, len(cubeface_images_src_list)):
        optical_flow_cubemap = optical_flow_method(cubeface_images_src_list[index], cubeface_images_tar_list[index])
        cubemap_face_of_list.append(optical_flow_cubemap)
    optical_flow_cubemap = proj_cm.cubemap2erp_flow(cubemap_face_of_list, erp_image_height, padding_size_cubemap, src_erp_image, tar_erp_image, wrap_around=True)
  
    # 3) accumulate all-steps optical flow
    of_accu = projection.flow_rotate_endpoint(optical_flow_cubemap, erp_rot_mat.T)

    if debug_output_dir is not None:
        debug_save_of(of_accu, debug_output_dir + "pano_of_0_ico_of_accu")
        
    return of_accu



def pano_of_1(src_erp_image, tar_erp_image, optical_flow_method=None, debug_output_dir=None):
    """Compute the optical flow with multi-step and icosahedron projection.

    The multi-steps method, transfer the warped ERP image to next step.
    1) ERP images (I_{src}, I_{tar}) optical flow (of_{erp}) get the rotation, rotate the target image to generate {I_{tar}_{erp} for next step;
    2) CubeMap the (I_{src}, I_{tar}_{erp}) to estimate the optical flow and stitch to  of_{cubemap}, and warp the I_{tar}_{erp} to generate the I_{tar}_{cubemap}
    3) Ico the {I_{src}, I_{tar}_{cube}} to estimate the optical flow and stitch to of_{ico}, and warp the I_{tar}_{cube} to generate the I_{tar}_{ico};
    4) accumulate all the optical flow together.

    @see pano_of_0
    """
    pass
