import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import image_evaluate
import image_io

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


"""
 optical flow visualization;
reference 
https://github.com/tomrunia/OpticalFlow_Visualization
"""


def make_colorwheel():
    """
    # TODO change comment style
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    # TODO change comment style
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False, min_ratio=0.0, max_ratio=1.0):
    """ Expects a two dimensional flow image of shape.

    :param flow_uv: Flow UV image of shape [H,W,2]
    :type flow_uv: numpy
    :param clip_flow: Clip maximum of flow values. Defaults to None e.g. [-100, 100]
    :type clip_flow: float
    :param convert_to_bgr: Convert output image to BGR. Defaults to False.
    :type convert_to_bgr: bool, optional
    :return: Flow visualization image of shape [H,W,3]
    :rtype: numpy
    """
    # get the clip range
    if min_ratio !=0 and max_ratio!=1.0:
        clip_flow = image_evaluate.get_min_max(flow_uv, min_ratio, max_ratio)

    # u_min,u_max = image_evaluate.get_min_max(flow_uv[:,:,0], min_ratio, max_ratio)
    # v_min,v_max = image_evaluate.get_min_max(flow_uv[:,:,1], min_ratio, max_ratio)
    # log.info("optical flow U range are [{}, {}], V range are [{}, {}]".format(u_min, u_max, v_min,v_max))

    # visualize optical flow
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, clip_flow[0], clip_flow[1])
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_value_to_color(flow_uv, output_path=None, min_ratio=0.0, max_ratio=1.0, visual_colormap = "RdPu"):
    """ Visualize U,V and show the bar.

    :param flow_uv: optical flow. [height, width, 2]
    :type flow_uv: numpy
    """
    # get vmin and vmax
    vmin_, vmax_ = image_evaluate.get_min_max(flow_uv, min_ratio, max_ratio)

    # draw image
    figure, axes = plt.subplots(2)

    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    # add sub caption
    axes[0].set_title("Optical Flow (U)")
    im = axes[0].imshow(flow_uv[:, :, 0], cmap=cm.get_cmap(visual_colormap), vmin=vmin_, vmax=vmax_)

    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    # add sub caption
    axes[1].set_title("Optical Flow (V)")
    im = axes[1].imshow(flow_uv[:, :, 1], cmap=cm.get_cmap(visual_colormap), vmin=vmin_, vmax=vmax_)

    figure.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(figure)


def flow_max_min_visual(flow_data, output_path=None,  min_ratio=0.1, max_ratio=0.9):
    """ Visualize the max and min of the optical flow.
    """
    # get vmin and vmax
    u_vmin_, u_vmax_ = image_evaluate.get_min_max(flow_data[:, :, 0], 0.0, 1.0)
    v_vmin_, v_vmax_ = image_evaluate.get_min_max(flow_data[:, :, 1], 0.0, 1.0)
    u_ratio_vmin_, u_ratio_vmax_ = image_evaluate.get_min_max(flow_data[:, :, 0], min_ratio, max_ratio)
    v_ratio_vmin_, v_ratio_vmax_ = image_evaluate.get_min_max(flow_data[:, :, 1], min_ratio, max_ratio)
    # new_u = np.where(flow_data[:,:,0] < vmin_, flow_data[:,:,0], 0)
    # new_u = np.where(flow_data[:,:,0] > vmax_, flow_data[:,:,0], vmin_)
    from skimage.morphology import dilation, square


    def image_process(flow_data):
        dilation_square_size = 10
        return dilation(flow_data, square(dilation_square_size))
        # return flow_data

    figure, axes = plt.subplots(2, 2)
    # u min
    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 0].get_yaxis().set_visible(False)
    axes[0, 0].set_title("Optical Flow (U) Min")
    flow_data_u_min = np.where(flow_data[:, :, 0] < u_ratio_vmin_, -flow_data[:, :, 0] + u_ratio_vmin_, 0)
    # im = axes[0, 0].imshow(flow_data_u_min, cmap=plt.get_cmap("Greys") )#, vmin=0, vmax=u_ratio_vmin_)
    im = axes[0, 0].imshow(image_process(flow_data_u_min), cmap=plt.get_cmap("Greys"))  # , vmin=0, vmax=u_ratio_vmin_)
    figure.colorbar(im, ax=axes[0, 0])#.ravel().tolist())
    # u max
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 1].get_yaxis().set_visible(False)
    axes[0, 1].set_title("Optical Flow (U) Max")
    flow_data_u_max = np.where(flow_data[:, :, 0] > u_ratio_vmax_, flow_data[:, :, 0] - u_ratio_vmax_, 0)
    im = axes[0, 1].imshow(image_process(flow_data_u_max), cmap=plt.get_cmap("Greys"))  # , vmin=u_ratio_vmax_, vmax=u_vmax_)
    figure.colorbar(im, ax=axes[0, 1])#.ravel().tolist())
    # v min
    axes[1, 0].get_xaxis().set_visible(False)
    axes[1, 0].get_yaxis().set_visible(False)
    axes[1, 0].set_title("Optical Flow (V) Min")
    flow_data_v_min = np.where(flow_data[:, :, 1] < v_ratio_vmin_, -flow_data[:, :, 1] + v_ratio_vmin_, 0)
    im = axes[1, 0].imshow(image_process(flow_data_v_min), cmap=plt.get_cmap("Greys"))  # , vmin=v_vmin_, vmax=v_ratio_vmin_)
    figure.colorbar(im, ax=axes[1, 0])#.ravel().tolist())
    # v max
    axes[1, 1].get_xaxis().set_visible(False)
    axes[1, 1].get_yaxis().set_visible(False)
    axes[1, 1].set_title("Optical Flow (V) Max")
    flow_data_v_max = np.where(flow_data[:, :, 1] > v_ratio_vmax_, flow_data[:, :, 1] - v_ratio_vmax_, 0)
    im = axes[1, 1].imshow(image_process(flow_data_v_max), cmap=plt.get_cmap("Greys"))  # , vmin=v_ratio_vmax_, vmax=v_vmax_)
    figure.colorbar(im, ax=axes[1, 1])#.ravel().tolist())

    # figure.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(figure)
    # flow_data_max = np.where(flow_data > vmax_, flow_data, 0)
    # image_io.image_show(dilation(flow_data_min[:,:,0], square(35)))
    # image_io.image_show(dilation(flow_data_min[:,:,1], square(35)))
    # flow_value_to_color(flow_data)
