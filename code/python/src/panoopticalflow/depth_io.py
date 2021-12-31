import os
import sys
import re
from struct import unpack

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from . import image_evaluate
from .logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def create_depth_mask(depth_map, output_filepath=None,  threshold=0.0):
    """Create the unavailable pixel mask from depth map.

    The unavailable pixel is the value less than the threshold.
    In the mask the available pixel label is no 0, unavailable pixel label is 0.

    :param depth_map: The original depth map, shape is [hight, width]
    :type depth_map: numpy
    :param threshold: Threshold, defaults to 0.0
    :type threshold: float, optional
    """
    pixel_mask = np.where(depth_map < threshold, 0, 65535)

    if output_filepath is not None:
        img = Image.fromarray(pixel_mask.astype(np.uint8))
        img.save(output_filepath, compress_level=0)

    return pixel_mask


def depth2disparity(depth_map, baseline=1.0, focal=1.0):
    """Convert the depth map to disparity map.

    :param depth_map: depth map data
    :type depth_map: numpy
    :param baseline: [description], defaults to 1
    :type baseline: float, optional
    :param focal: [description], defaults to 1
    :type focal: float, optional
    :return: disparity map data, 
    :rtype: numpy
    """
    no_zeros_index = np.where(depth_map != 0)
    disparity_map = np.full(depth_map.shape, np.Inf, np.float64)
    disparity_map[no_zeros_index] = (baseline * focal) / depth_map[no_zeros_index]
    return disparity_map


def depth_visual_save(depth_data, output_path=None, min_ratio=0.05, max_ratio=0.95, visual_colormap = "RdPu"):
    """save the visualized depth map to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image, if is None return rendering result.
    :type output_path: str
    :return:
    :rtype: numpy
    """
    dapthe_data_temp = depth_data.astype(np.float64)
    vmin_, vmax_ = image_evaluate.get_min_max(depth_data, min_ratio, max_ratio)
    if min_ratio != 0 or max_ratio != 1.0:
        log.warn("clamp the depth value form [{},{}] to [{},{}]".format(np.amin(depth_data), np.amax(depth_data), vmin_, vmax_))

    # draw image
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1, wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    im = ax.imshow(dapthe_data_temp, cmap=cm.get_cmap(visual_colormap), vmin=vmin_, vmax=vmax_)
    #im = ax.imshow(disparity_data, cmap=cm.coolwarm)
    cbar = ax.figure.colorbar(im, ax=ax)
    result_image = None
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    else:
        fig.canvas.draw()  # draw the renderer
        w, h = fig.canvas.get_width_height()  # Get the RGBA buffer from the figure
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (w, h, 3)
        image = Image.frombytes("RGB", (w, h), buf.tostring())
        result_image = np.asarray(image)
    # plt.close(fig)
    plt.clf()
    plt.cla()
    plt.close("all")
    return result_image


def read_bin(binary_file_path, height, width):
    """Load depth value form binary file.
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    depth_data = xbash.reshape(height, width, 1)
    return depth_data


def read_png_int8(png_file_path):
    """Read depth map from png file.
    """
    depth_data = np.array(Image.open(png_file_path))

    channels_r = depth_data[:, :, 0].astype(np.float32)
    channels_g = depth_data[:, :, 1].astype(np.float32)
    channels_b = depth_data[:, :, 2].astype(np.float32)

    depth_map = channels_r + channels_g * 256 + channels_b * 256 * 256
    depth_map = depth_map / 65536.0

    return depth_map


def write_png_int8(depth_data, png_file_path):
    """Write depth map to 24bit three channel png file.

    :param depth_data: numpy
    :type depth_data: depth map.
    :param png_file_path: str
    :type png_file_path: output file path
    """
    depth_value = (depth_data * 65536.0).astype(int)

    channels_r = np.remainder(depth_value, 256)

    depth_value = (depth_value / 256).astype(int)
    channels_g = np.remainder(depth_value, 256)

    depth_value = (depth_value / 256).astype(int)
    channels_b = np.remainder(depth_value, 256)

    png_data = np.stack((channels_r, channels_g, channels_b), axis=2)

    img = Image.fromarray(png_data.astype(np.uint8))
    img.save(png_file_path, compress_level=0)


def write_png_int16(png_file_path, depth):
    """Write depth map to 16bit single channel png file.

    :param png_file_path: png file path
    :type png_file_path: str
    :param depth: depth map data
    :type depth: numpy
    """
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*2))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    img = Image.fromarray(out.astype(np.uint16))
    img.save(png_file_path, compress_level=0)


def read_dpt(dpt_file_path):
    """Read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


def write_dpt(depth_data, dpt_file_path):
    """Save the depth map from a .dpt file (Sintel format).

    :param depth_data: the depth map's data [height, width]
    :type depth_data: numpy
    :param dpt_file_path: dpt file path
    :type dpt_file_path: str
    """
    if not len(depth_data.shape) == 2:
        log.error("the depth dimension should be 2.")
        # raise RuntimeError("the depth dimension is not 1.")

    width = np.shape(depth_data)[1]
    height = np.shape(depth_data)[0]

    with open(dpt_file_path, 'wb') as file_handle:
        file_handle.write("PIEH".encode())
        np.array(width).astype(np.int32).tofile(file_handle)
        np.array(height).astype(np.int32).tofile(file_handle)
        depth_data.astype(np.float32).tofile(file_handle)


def read_pfm(path):
    """Read pfm file.

    :param path: the PFM file's path.
    :type path: str
    :return: the depth map array and scaler of depth
    :rtype: tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            log.error("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            log.error("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write depth data to pfm file.

    :param path: pfm file path
    :type path: str
    :param image: depth data
    :type image: numpy
    :param scale: Scale, defaults to 1
    :type scale: int, optional
    """
    if image.dtype.name != "float32":
        #raise Exception("Image dtype must be float32.")
        log.warn("The depth map data is {}, convert to float32 and save to pfm format.".format(image.dtype.name))

    image = np.flipud(image)

    color = None
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        log.error("Image must have H x W x 3, H x W x 1 or H x W dimensions.")
        # raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    with open(path, "wb") as file:
        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))
        endian = image.dtype.byteorder
        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale
        file.write("%f\n".encode() % scale)
        image.tofile(file)
