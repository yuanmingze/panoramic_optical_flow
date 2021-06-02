import os
import sys

# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.abspath(__file__))
dir_root = os.path.dirname(dir_scripts)
dir_utility = os.path.join(dir_root, "utility")
dir_replica360 = os.path.join(dir_root, "replica360")

sys.path.append(dir_scripts)
sys.path.append(dir_root)
sys.path.append(dir_utility)
sys.path.append(dir_replica360)