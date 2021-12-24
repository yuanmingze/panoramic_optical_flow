
import pathlib
import os
import shutil
from random import shuffle

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


def list_files(folder_path, extension):
    """ List all file with specified extension name.

    :param folder_path: the files root dir.
    :type folder_path: str
    :param extension: file extension name, eg al. .jpg.
    :type extension: str
    """
    files_list = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            files_list.append(file)
    return files_list


def copy_replace(src_filepath, tar_filepath, replace_list = None):
    """ Copy file and replace words.

    :param src_filepath: the source file path.
    :type src_filepath: str
    :param tar_filepath: the target file path.
    :type tar_filepath: str
    :param replace_list: the word list need to replace.
    :type replace_list: dict
    """    
    if replace_list is None:
        log.info("The replace words list is empty. Copy file directly.")
        shutil.copy(src_filepath, tar_filepath)

    with open(src_filepath, 'r') as file :
        filedata = file.read()

    for old_word in replace_list:
        new_word = replace_list[old_word]
        filedata = filedata.replace(old_word, new_word)
    
    with open(tar_filepath, 'w') as file:
        file.write(filedata)