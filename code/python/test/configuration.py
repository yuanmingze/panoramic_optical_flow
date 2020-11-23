import os
import sys

# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dir_root = os.path.dirname(dir_scripts) # parent dir
print(dir_scripts)
sys.path.append(dir_scripts)
sys.path.append(os.path.join(dir_scripts, "utility"))

# setting the test data path
TEST_data_root_dir = "../../../data/"
