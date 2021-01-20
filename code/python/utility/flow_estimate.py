import cv2

from .logger import Logger
import image_io
import projection

import projection_cubemap as proj_cm
import projection_icosahedron as proj_ico

log = Logger(__name__)
log.logger.propagate = False


def DIS(image_src, image_tar):
    """Compute the DIS flow.

    :param image_src: The optical flow source image.
    :type image_src: numpy
    :param image_tar: The optical flow target image.
    :type image_tar: numpy
    :return: the optical flow.
    :rtype: numpy
    """
    # the image including alpha channel
    if image_src.shape[2] == 4:
        image_src = image_src[:, :, :3]
    if image_tar.shape[2] == 4:
        image_tar = image_tar[:, :, :3]

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


def multi_step_DIS(src_erp_image, tar_erp_image, optical_flow_method=None):
    """Compute the optical flow with mulit-step and icosahedron projection.

    :param src_erp_image: the source ERP image data.
    :type src_erp_image: numpy
    :param tar_erp_image: the target ERP image data.
    :type tar_erp_image: numpy
    :param optical_flow_method: the optical flow estimation function.
    :type: function
    :return: the optical flow from src image to tar image.
    :rtype: numpy
    """
    if optical_flow_method == None:
        optical_flow_method = DIS

    erp_image_height = src_erp_image.shape[0]

    # 0) compute flow in ERP image & wrap image
    log.debug("compute ERP image flow")
    optical_flow_erp = optical_flow_method(src_erp_image, tar_erp_image)
    tar_erp_image_rot_erp, image_rotation_erp = projection.image_align(tar_erp_image, optical_flow_erp)

    # 1) compute flow with cubemap projection & warp target image
    log.debug("compute cubemap projection image flow")
    # 1-1) erp image to cube map
    padding_size_cubemap = 0.1
    cubeface_images_src_list = proj_cm.erp2cubemap_image(src_erp_image, padding_size_cubemap)
    cubeface_images_tar_list = proj_cm.erp2cubemap_image(tar_erp_image_rot_erp, padding_size_cubemap)
    cubemap_face_of_list = []
    for index in range(0, len(cubeface_images_src_list)):
        optical_flow_cubemap = optical_flow_method(cubeface_images_src_list[index], cubeface_images_tar_list[index])
        cubemap_face_of_list.append(optical_flow_cubemap)
    optical_flow_cubemap = proj_cm.cubemap2erp_flow(cubemap_face_of_list, erp_image_height, padding_size=padding_size_cubemap)

    # 1-2) warp target image
    tar_erp_image_rot_cubemap, image_rotation_cubemap = projection.image_align(tar_erp_image_rot_erp, optical_flow_cubemap)

    # 2) compute flow with icosahedron projection & warp image
    log.debug("compute icosahedron projection image flow")
    # 2-1) erp image to cube map
    padding_size_ico = 0.1
    icoface_images_src_list = proj_ico.erp2ico_image(src_erp_image, padding_size_ico)
    icoface_images_tar_list = proj_ico.erp2ico_image(tar_erp_image_rot_cubemap, padding_size_ico)
    ico_face_of_list = []
    for index in range(0, len(icoface_images_src_list)):
        optical_flow_ico = optical_flow_method(icoface_images_src_list[index], icoface_images_tar_list[index])
        ico_face_of_list.append(optical_flow_ico)
    optical_flow_ico = proj_cm.cubemap2erp_flow(ico_face_of_list, erp_image_height, padding_size=padding_size_ico)

    # 2-2) warp target image
    tar_erp_image_rot_ico, image_rotation_ico = projection.image_align(tar_erp_image_rot_cubemap, optical_flow_ico)

    # 3) accumulate all-steps optical flow
    of_ico2cub = projection.flow_accumulate_endpoint(optical_flow_ico, image_rotation_ico)
    of_accu = projection.flow_accumulate_endpoint(of_ico2cub, image_rotation_cubemap)

    return of_accu
