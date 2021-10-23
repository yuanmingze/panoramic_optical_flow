import os
import sys

# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dir_root = os.path.dirname(dir_scripts) # parent dir
# print(dir_scripts)
sys.path.append(dir_scripts) #  code/python folder
# sys.path.append(os.path.join(dir_scripts, "utility")) # code/python/utility folder

# setting the test data path
TEST_data_root_dir = "../../../data/"
TEMP_data_root_dir = "../../../data/temp/"
