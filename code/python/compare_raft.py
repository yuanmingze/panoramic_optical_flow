import sys
sys.path.append('core')
from utils.utils import InputPadder
from utils import flow_viz
from raft import RAFT
from PIL import Image
import torch
import numpy as np
import glob
import cv2
import os
import argparse
import sys
import struct
import re
import pathlib
from struct import pack, unpack

INDEX_MAX = sys.maxsize
DEVICE = 'cuda'


def scene_of_folder(root_dir):
    """
    """
    # search all rgb image
    index_digit_re = r"\d{4}"
    index_digit_format = r"{:04d}"

    flow_fw_fne_str = index_digit_re + r"_opticalflow_forward.flo"
    flow_bw_fne_str = index_digit_re + r"_opticalflow_backward.flo"

    rgb_image_fne = re.compile(index_digit_re + r"_rgb.jpg")
    flow_fw_fne = re.compile(flow_fw_fne_str)
    flow_bw_fne = re.compile(flow_bw_fne_str)

    # scan all file to get the index, and load image and of list
    of_forward_list = {}
    of_backward_list = {}
    image_list = {}
    min_index = INDEX_MAX
    max_index = -1

    for item in pathlib.Path(root_dir).iterdir():
        index_number = re.search(index_digit_re, item.name).group()
        if index_number is None:
            continue

        index_number = int(index_number)
        if index_number > max_index:
            max_index = index_number
        if index_number < min_index:
            min_index = index_number

        if not rgb_image_fne.match(item.name) is None:
            image_list[index_number] = item.name
        elif not flow_fw_fne.match(item.name) is None:
            of_forward_list[index_number] = item.name
        elif not flow_bw_fne.match(item.name) is None:
            of_backward_list[index_number] = item.name

    return min_index, max_index, image_list, of_forward_list, of_backward_list


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


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


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


# def viz(img, flo):

#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)
#     return img_flo

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    # 0) initial context
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    root_dir = args.path
    output_dir = root_dir + "../raft/"

    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("---start---")

    # 1) load data information
    min_index, max_index, image_files, of_forward_list, of_backward_list = scene_of_folder(root_dir)

    # 2) estimate optical flow
    with torch.no_grad():
        for index in range(min_index, max_index + 1):

            if index % 1 == 0:
                print("{}: {}".format(index, image_files[index]))

            index_next = index + 1
            if index_next > max_index:
                index_next = min_index + max_index - index

            index_previous = index - 1
            if index_previous < min_index:
                index_previous = max_index + min_index - index

            # load image a image pair
            # 2-0) inference optical flow forward
            images = load_image_list([root_dir + image_files[index], root_dir + image_files[index_next]])

            image1 = images[0, None]
            image2 = images[1, None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            writeFlowFile(flow_up, output_dir + of_forward_list[index])
            flow_vis = flow_viz.flow_to_image(flow_up)
            flow_vis = Image.fromarray(flow_vis)
            flow_vis.save((output_dir + of_forward_list[index]).replace(".flo", ".jpg"))

            # 2-1) inference optical flow backward
            images = load_image_list([root_dir + image_files[index], root_dir + image_files[index_previous]])

            image1 = images[0, None]
            image2 = images[1, None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            writeFlowFile(flow_up, output_dir + of_backward_list[index])
            flow_vis = flow_viz.flow_to_image(flow_up)
            flow_vis = Image.fromarray(flow_vis)
            flow_vis.save((output_dir + of_backward_list[index]).replace(".flo", ".jpg"))


if __name__ == '__main__':
    """
    e.g.
    python compare_raft.py --model=models/raft-things.pth --path=/mnt/sda1/workdata/opticalflow_data/replica_360/${DATASET_NAME}/replica_seq_data/
    """

    dataset_list = ["apartment_0", "hotel_0", "office_0", "office_4", "room_0", "room_1"]

    for dataset_name in dataset_list:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        # args = parser.parse_args()
        flo_path = "/mnt/sda1/workdata/opticalflow_data/replica_360/"+dataset_name+"/replica_seq_data/"
        args = parser.parse_args(['--model', 'models/raft-things.pth', '--path', flo_path])
        demo(args)
