
import pathlib
import os

import ipdb

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def dir_make(directory):
    """
    check the existence of directory, if not mkdir
    :param directory: the directory path
    :type directory: str
    """
    # check
    if isinstance(directory, str):
        directory_path = pathlib.Path(directory)
    elif isinstance(directory, pathlib.Path):
        directory_path = directory
    else:
        log.warn("Directory is neither str nor pathlib.Path {}".format(directory))
        return
    # create folder
    if not directory_path.exists():
        directory_path.mkdir()
    else:
        log.info("Directory {} exist".format(directory))


def dir_grep(dir_path, postfix):
    """Find all files in a directory with extension.

    :param dir_path: folder path.
    :type dir_path: str
    :param postfix: extension, e.g. ".txt"
    :type postfix: str
    """
    file_list = []
    for file in os.listdir(dir_path):
        if file.endswith(postfix):
            file_list.append(file)
    file_list.sort()
    return file_list
