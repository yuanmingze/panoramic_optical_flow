import os
import sys
import re
from struct import pack, unpack

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def depth_visual_save(depth_data, output_path):
    """save the visualized depth map to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image.
    :type output_path: str
    """
    dapthe_data_temp = depth_data.astype(np.float64)

    # draw image
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1, wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    im = ax.imshow(dapthe_data_temp, cmap=cm.jet)
    #im = ax.imshow(disparity_data, cmap=cm.coolwarm)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def depth_visual(depth_data):
    """
    visualize the depth map
    """
    min = np.min(depth_data)
    max = np.max(depth_data)
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cmap = plt.get_cmap('jet')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return (m.to_rgba(depth_data)[:, :, :3] * 255).astype(np.uint8)


def read_bin(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    depth_data = xbash.reshape(height, width, 1)
    return depth_data


def read_png_int8(png_file_path):
    """
    read depth map from png file.
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

    :param depth_data: [description]
    :type depth_data: [type]
    :param png_file_path: [description]
    :type png_file_path: [type]
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


def write_png_int16(png_file_path, depth,):
    """Write depth map to 16bit single channel png file.

    :param png_file_path: png file path
    :type png_file_path: str
    :param depth: depth map data
    :type depth: numpy
    :param bits: the byte number store single pixel depth map, defaults to 2
    :type bits: int, optional
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
    """read depth map from *.dpt file.

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

    :param depth_data: the depth map's data
    :type depth_data: numpy
    :param dpt_file_path: dpt file path
    :type dpt_file_path: str
    """
    if not len(depth_data.shape) == 2:
        log.error("the depth dimension should be 2.")
        # raise RuntimeError("the depth dimension is not 1.")

    width = np.shape(depth_data)[0]
    height = np.shape(depth_data)[1]

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
            raise Exception("Not a PFM file: " + path)

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
