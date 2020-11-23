import os
from struct import pack, unpack

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.cm as cm


def depth_visual(depth_data):
    """
    visualize the depth map
    """
    min = np.min(depth_data)
    max = np.max(depth_data)
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cmap = plt.get_cmap('jet')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return (m.to_rgba(depth_data)[:,:,:3] * 255).astype(np.uint8)


def read_bin(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    depth_data = xbash.reshape(height, width, 1)
    # imgplot = plt.imshow(data[:,:,0])
    # plt.show()
    return depth_data


def write_bin(depth_data, binary_file_path):
    """
    write the depth map to bin file
    """
    raise RuntimeError("do not implement")


def read_png(png_file_path):
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


def write_png(depth_data, png_file_path):
    """
    output the depth map to png file, 
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


def read_dpt(dpt_file_path):
    """
    read depth map from *.dpt file.
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

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
    """
    Reads a .dpt file (Sintel format).
    """
    if not len(np.shape(data)) == 1:
        raise RuntimeError("the depth dimension is not 1.")

    width = np.shape(depth_data)[0]
    height = np.shape(depth_data)[1]

    with open(dpt_file_path, 'wb') as file_handle:
        file_handle.write("PIEH".encode())
        np.array(width).astype(np.int32).tofile(file_handle)
        np.array(height).astype(np.int32).tofile(file_handle)
        depth_data.astype(np.float32).tofile(file_handle)
