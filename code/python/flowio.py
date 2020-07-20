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
