import configuration as config

from utility import flow_estimate
from utility import image_io
from utility import flow_io
from utility import flow_vis
from utility import flow_warp

from utility import flow_evaluate

import os


def test_opticalflow_metric(erp_flo_gt_filepath, erp_flo_eva_filepath, erp_flo_mask_filepath, erp_flo_output_dir):
    flo_gt = flow_io.read_flow_flo(erp_flo_gt_filepath)
    flo_eva = flow_io.read_flow_flo(erp_flo_eva_filepath)
    flo_mask = flow_io.read_mask(erp_flo_mask_filepath)

    print("AAE: {}".format(flow_evaluate.AAE(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)))
    print("EPE: {}".format(flow_evaluate.EPE(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)))
    print("RMS: {}".format(flow_evaluate.RMSE(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)))


    image_io.image_show()
    flow_evaluate.AAE_mat(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)
    flow_evaluate.EPE_mat(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)
    flow_evaluate.RMSE_mat(erp_flo_gt_filepath, erp_flo_eva_filepath, flo_mask)

if __name__ == "__main__":
    erp_flo_gt_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    erp_flo_eva_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    erp_flo_mask_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    erp_flo_output_dir = config.TEST_data_root_dir + "replica_360/apartment_0/"

    test_opticalflow_metric(erp_flo_gt_filepath, erp_flo_eva_filepath, erp_flo_mask_filepath, erp_flo_output_dir)
