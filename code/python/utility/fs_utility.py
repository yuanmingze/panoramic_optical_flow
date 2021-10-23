
import pathlib
import os

from .logger import Logger

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


def dir_ls(dir_path, postfix = None):
    """Find all files in a directory with extension.

    :param dir_path: folder path.
    :type dir_path: str
    :param postfix: extension, e.g. ".txt", if it's none list all folders name.
    :type postfix: str
    """
    file_list = []
    for file_name in os.listdir(dir_path):
        if os.path.isdir(dir_path + "/" + file_name) and postfix is None:
            file_list.append(file_name)
        elif postfix is not None:
            if file_name.endswith(postfix):
                file_list.append(file_name)
    file_list.sort()
    return file_list


def dir_rm(dir_path):
    """Deleting folders recursively.

    :param dir_path: The folder path.
    :type dir_path: str
    """
    directory = pathlib.Path(dir_path)
    if not directory.exists():
        log.warn("Directory {} do not exist".format(dir_path))
        return
    for item in directory.iterdir():
        if item.is_dir():
            dir_rm(item)
        else:
            item.unlink()
    directory.rmdir()

