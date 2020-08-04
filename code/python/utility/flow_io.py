import os
import numpy as np
from struct import pack, unpack


def readFlowFile(fname):
    '''
    args
        fname (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('readFlowFile: fname %s should have extension ''.flo''' % fname)

    try:
        fid = open(fname, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', fname)

    tag     = unpack('f', fid.read(4))[0]
    width   = unpack('i', fid.read(4))[0]
    height  = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % fname)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (fname, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (fname, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow

def writeFlowFile(img, fname):
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

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



def of_ph2pano(optical_flow, optical_flow_new, of_warp_around_threshold = 0.5):
    """
    convert the pinhole optical flow to panoramic optical flow.
    
    basic suppose: if optical flow in 
    :param:
    :param:
    """
    of_image_size = np.shape(optical_flow)
    image_height = of_image_size[0]
    image_width = of_image_size[1]
    image_channels = of_image_size[2]
    of_warp_around_threshold = image_width * of_warp_around_threshold

    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = optical_flow[:, :, 0]
    of_forward_y = optical_flow[:, :, 1]

    optical_flow_new[:] = optical_flow

    warp_around_idx = np.where(of_forward_x > of_warp_around_threshold)
    optical_flow_new[:, :, 0][warp_around_idx] = of_forward_x[warp_around_idx] - image_width

    warp_around_idx = np.where(of_forward_x < -of_warp_around_threshold)
    optical_flow_new[:, :, 0][warp_around_idx] = of_forward_x[warp_around_idx] + image_width


def of_pano2ph(optical_flow, optical_flow_new):
    """
    panoramic optical flow to pinhole optical flow.

    process the panorama optical flow, change the warp around to normal optical flow.
    :param: the panoramic optical flow
    :param: the optical flow processed warp around
    """
    of_image_size = np.shape(optical_flow)
    image_height = of_image_size[0]
    image_width = of_image_size[1]
    image_channels = of_image_size[2]

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx_tar, y_idx_tar = np.meshgrid(x_idx_arr, y_idx_arr)

    of_forward_x = optical_flow[:, :, 0]
    of_forward_y = optical_flow[:, :, 1]
    x_idx = (x_idx_tar + of_forward_x + 0.5).astype(np.int)
    y_idx = (y_idx_tar + of_forward_y + 0.5).astype(np.int)

    # 1) process the warp around
    optical_flow_new[:] = optical_flow

    # process optical flow x
    x_idx_outrange_idx = np.where(x_idx >= image_width)
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx[x_idx_outrange_idx] - image_width
    x_idx_outrange_idx = np.where(x_idx < 0)
    optical_flow_new[:, :, 0][x_idx_outrange_idx] = x_idx[x_idx_outrange_idx] + image_width

    # process optical flow y
    y_idx_outrange_idx = np.where(y_idx >= image_height)
    optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx[y_idx_outrange_idx] - image_height
    y_idx_outrange_idx = np.where(y_idx < 0)
    optical_flow_new[:, :, 1][y_idx_outrange_idx] = y_idx[y_idx_outrange_idx] + image_height


def load_of_bin(binary_file_path, height, width, visual_enable=True):
    """
    load depth value form binary file, replica360 generated 
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(width, height, 4)

    if visual_enable:
        cmap = plt.get_cmap('rainbow')
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

        images = []
        axs[0].set_title('optical flow x')
        images.append(axs[0].imshow(data[:, :, 0], cmap=cmap))

        axs[1].set_title('optical flow y')
        images.append(axs[1].imshow(data[:, :, 1], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    if visual_enable:
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))

        images = []
        axs.set_title('optical flow x')
        images.append(axs.imshow(data[:, :, 0], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    if visual_enable:
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))

        images = []
        axs.set_title('optical flow y')
        images.append(axs.imshow(data[:, :, 1], cmap=cmap))

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()

    return data[:, :, 0:2], data[:,:,2:4]
