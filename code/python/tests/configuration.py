import os
import sys

# to import module in sibling folders
python_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(dir_scripts + "/src") #  code/python/src 
sys.path.append(python_root_dir + "/src")
# sys.path.append(python_root_dir)


# setting the test data path
TEST_data_root_dir = "../../../data/"
TEMP_data_root_dir = "../../../data/temp/"
