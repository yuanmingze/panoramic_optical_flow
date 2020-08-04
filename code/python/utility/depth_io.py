import numpy as np


def load_depth_bin(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(height, width, 1)
    # imgplot = plt.imshow(data[:,:,0])
    # plt.show()
    return data


def load_depth_bin(binary_file_path, height, width):
    """
    load depht value form binary file
    """
    xbash = np.fromfile(binary_file_path, dtype='float32')
    data = xbash.reshape(height, width, 1)
    return data