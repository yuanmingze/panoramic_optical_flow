import argparse
import os

from panoopticalflow.visualization import vis_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=str, help='the task index')

    args = parser.parse_args()

    test_list = []
    test_list.append(args.task)

    cur_dir = os.getcwd()

    if args.task == "vis":
        print("visualize folder {}".format(cur_dir))
        vis_dir(cur_dir)
