import sys
sys.path.append('core')

import argparse
import os
from struct import pack, unpack

import pathlib
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



# dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dir_root = os.path.dirname(dir_scripts) # parent dir
dir_scripts = "D:/workspace_windows/panoramic_optical_flow/code/python/"
print(dir_scripts)
sys.path.append(dir_scripts)
sys.path.append(os.path.join(dir_scripts, "test"))
sys.path.append(os.path.join(dir_scripts, "utility"))

from utility import fs_utility
from main import ReplicaPanoDataset

DEVICE = 'cuda'

def write_flow_flo(img, fname):
    """
    
    """
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

    fid = None
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


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    img_flo = np.concatenate([flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)



def replica_test(replica_dataset, args):
    """Get the our and DIS's result in replica. """

    opticalflow_mathod = "raft"  # our, dis

    dataset_dirlist = replica_dataset.dataset_circ_dirlist + replica_dataset.dataset_line_dirlist

    # initial RAFT
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        # 1) iterate each 360 image dataset
        for pano_image_folder in dataset_dirlist:
            print("processing the data folder {}".format(pano_image_folder))
            # input dir
            input_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_data_dir + "/"
            # input index
            if pano_image_folder.find("line") != -1:
                pano_start_idx = replica_dataset.line_start_idx
                pano_end_idx = replica_dataset.line_end_idx
            elif pano_image_folder.find("circ") != -1:
                pano_start_idx = replica_dataset.circle_start_idx
                pano_end_idx = replica_dataset.circle_end_idx
            else:
                print("{} folder naming is wrong".format(pano_image_folder))

            # output folder
            output_pano_filepath = replica_dataset.pano_dataset_root_dir + pano_image_folder + "/" + replica_dataset.pano_output_dir
            output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
            fs_utility.dir_make(output_pano_filepath)
            fs_utility.dir_make(output_dir)

            for pano_image_idx in range(pano_start_idx, pano_end_idx):
                for forward_of in [True, False]:
                    # 0) load image to CPU memory
                    if forward_of:
                        src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                        tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx + 1)
                        optical_flow_filepath = replica_dataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    else:
                        src_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                        tar_erp_image_filepath = replica_dataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx - 1)
                        optical_flow_filepath = replica_dataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)


                    if pano_image_idx % 2 == 0:
                        print("{} Flow Method: {}\n{}\n{}".format(opticalflow_mathod, pano_image_idx, src_erp_image_filepath, tar_erp_image_filepath))
                    # 1) estimate optical flow
                    src_erp_image = load_image(input_filepath + src_erp_image_filepath)
                    tar_erp_image = load_image(input_filepath + tar_erp_image_filepath)

                    padder = InputPadder(src_erp_image.shape)
                    src_erp_image, tar_erp_image = padder.pad(src_erp_image, tar_erp_image)

                    flow_low, flow_up = model(src_erp_image, tar_erp_image, iters=20, test_mode=True)

                    # output optical flow image
                    result_opticalflow_filepath = output_dir + optical_flow_filepath
                    write_flow_flo(flow_up[0].permute(1,2,0).cpu().numpy(), result_opticalflow_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # python bmvc2021.py --model=models/raft-things.pth 

    replica_test(ReplicaPanoDataset, args)
    
