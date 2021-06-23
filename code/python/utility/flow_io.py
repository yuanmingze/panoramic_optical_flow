import os
from struct import pack, unpack

import numpy as np
import image_io
from logger import Logger

"""
 *.flo file read and wirte;
"""


log = Logger(__name__)
log.logger.propagate = False


def flow_read(file_path):
    """Load the optical flow from file.
    :param file_path: the optical flow file path
    :type files: str
    """
    if not os.path.exists(file_path):
        log.error("file {} do not exist.".format(file_path))

    # get the file format from the extension name
    _, format_str = os.path.splitext(file_path)
    if format_str == ".flo":
        return read_flow_flo(file_path)
    elif format_str == ".floss":
        return read_flow_floss(file_path)


def flow_write(flow_data, file_path):
    """[summary]

    :param flow_data: the optical flow data array, required size is (height, width, 2)
    :type flow_data: numpy
    :param file_path: the output file path.
    :type file_path: str
    """
    if os.path.exists(file_path):
        log.warn("file {} exist.".format(file_path))

    # get the file format from the extension name
    _, format_str = os.path.splitext(file_path)
    if format_str == ".flo":
        return write_flow_flo(flow_data, file_path)
    elif format_str == ".floss":
        return write_flow_floss(file_path)


# def readFlowFile(file_name):
def read_flow_flo(file_name):
    '''
    args
        file_name (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''
    if not os.path.exists(file_name):
        log.error("{} do not exist!".format(file_name))

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(file_name)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in file_name %s' % file_name)
    assert ext == '.flo', exit('readFlowFile: file_name %s should have extension ''.flo''' % file_name)

    try:
        fid = open(file_name, 'rb')
    except IOError:
        log.error('readFlowFile: could not open %s', file_name)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % file_name)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (file_name, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (file_name, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow


# def writeFlowFile(img, fname):
def write_flow_flo(img, fname):
    """
    
    """
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

    fid = None
    try:
        fid = open(fname, 'wb')
    except IOError:
        print('writeFlowFile: could not open %s', fname)

    # write the header
    # fid.write(TAG_STRING.encode(encoding='utf-8', errors='strict'))
    # code = unpack('f', bytes(TAG_STRING, 'utf-8'))[0]
    # fid.write(pack('f', code))
    fid.write(bytes(TAG_STRING, 'utf-8'))
    fid.write(pack('i', width))
    fid.write(pack('i', height))

    # arrange into matrix form
    tmp = np.zeros((height, width*nBands), np.float32)

    tmp[:, np.arange(width) * nBands] = img[:, :, 0]
    tmp[:, np.arange(width) * nBands + 1] = np.squeeze(img[:, :, 1])

    fid.write(bytes(tmp))

    fid.close()


# def readFlowFloss(fname):
def read_flow_floss(fname):
    '''
    args
        fname (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''
    if not os.path.exists(fname):
        raise RuntimeError("{} do not exist!".format(fname))

    TAG_STRING = "SHRT"  # check for this when READING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % fname)
    assert ext == '.floss', exit('readFlowFile: fname %s should have extension ''.flo''' % fname)

    try:
        fid = open(fname, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', fname)

    tag = unpack('4s', fid.read(4))[0].decode()
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_STRING, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % fname)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (fname, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (fname, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.int16)
    flow = flow.reshape(height, width, nBands)
    flow = flow / 8.0

    fid.close()

    return flow


# def writeFlowFloss(filename, uv, v=None):
def write_flow_floss(filename, uv, v=None):
    """ Write optical flow to file as format *.floss
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write("SHRT".encode())
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u * 8.0
    tmp[:, np.arange(width)*2 + 1] = v * 8.0
    tmp.astype(np.int16).tofile(f)
    f.close()


def read_flow_bin(binary_file_path, height, width):
    """
    load depth value form binary file, replica360 generated.
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(width, height, 4)
    return data[:, :, 0:2], data[:, :, 2:4]


def read_mask(mask_filepath):
    """Read the unavailable pixel mask from file.
    PNG file, available pixel is True (Not 0), unavailable pixel is Flase (0).

    :param mask_filepath: the file path.
    :type mask_filepath: str
    :return: the mask matrix.
    :rtype: numpy
    """
    _, file_extension = os.path.splitext(mask_filepath)
    mask_mat = None
    if file_extension == ".png":
        mask_mat = image_io.image_read(mask_filepath)
    else:
        log.warn("Do not support {} format mask".format(file_extension))

    return mask_mat
