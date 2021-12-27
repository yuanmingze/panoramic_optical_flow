import configuration as config

import image_io
import flow_io
import flow_evaluate


def test_opticalflow_metric(erp_flo_gt_filepath, erp_flo_eva_filepath, erp_flo_mask_filepath, erp_flo_output_dir):
    flo_gt = flow_io.read_flow_flo(erp_flo_gt_filepath)
    flo_eva = flow_io.read_flow_flo(erp_flo_eva_filepath)
    flo_mask = None #flow_io.read_mask(erp_flo_mask_filepath)

    print("AAE: {}".format(flow_evaluate.AAE(flo_gt, flo_eva, False, flo_mask)))
    print("EPE: {}".format(flow_evaluate.EPE(flo_gt, flo_eva, False, flo_mask)))
    print("RMS: {}".format(flow_evaluate.RMSE(flo_gt, flo_eva, False, flo_mask)))

    print("AAE Spherical: {}".format(flow_evaluate.AAE(flo_gt, flo_eva, True, flo_mask)))
    print("EPE Spherical: {}".format(flow_evaluate.EPE(flo_gt, flo_eva, True, flo_mask)))
    print("RMS Spherical: {}".format(flow_evaluate.RMSE(flo_gt, flo_eva, True, flo_mask)))

    min_ratio = 0.2
    max_ratio = 0.8
    print("AAE_mat: {}".format(erp_flo_output_dir + "aae_mat.jpg"))
    aae_mat, _ = flow_evaluate.AAE_mat(flo_gt, flo_eva, False, flo_mask)
    aae_mat_vis = image_io.visual_data(aae_mat, min_ratio, max_ratio)
    image_io.image_save(aae_mat_vis, erp_flo_output_dir + "aae_mat.jpg")

    print("AAE_Sph_mat: {}".format(erp_flo_output_dir + "aae_mat_sph.jpg"))
    aae_mat_sph, _ = flow_evaluate.AAE_mat(flo_gt, flo_eva, True, flo_mask)
    aae_mat_sph_vis = image_io.visual_data(aae_mat_sph, min_ratio, max_ratio)
    image_io.image_save(aae_mat_sph_vis, erp_flo_output_dir + "aae_mat_sph.jpg")

    print("EPE_mat: {}".format(erp_flo_output_dir + "epe_mat.jpg"))
    epe_mat, _ = flow_evaluate.EPE_mat(flo_gt, flo_eva,  False, flo_mask)
    epe_mat_vis = image_io.visual_data(epe_mat, min_ratio, max_ratio)
    image_io.image_save(epe_mat_vis, erp_flo_output_dir + "epe_mat.jpg")

    print("EPE_Sph_mat: {}".format(erp_flo_output_dir + "epe_mat_sph.jpg"))
    epe_mat_sph, _ = flow_evaluate.EPE_mat(flo_gt, flo_eva,  True, flo_mask)
    epe_mat_sph_vis = image_io.visual_data(epe_mat_sph, min_ratio, max_ratio)
    image_io.image_save(epe_mat_sph_vis, erp_flo_output_dir + "epe_mat_sph.jpg")

    print("RMS_mat: {}".format(erp_flo_output_dir + "rms_mat.jpg"))
    rms_mat, _ = flow_evaluate.RMSE_mat(flo_gt, flo_eva,  False, flo_mask)
    rms_mat_vis = image_io.visual_data(rms_mat, min_ratio, max_ratio)
    image_io.image_save(rms_mat_vis, erp_flo_output_dir + "rms_mat.jpg")

    print("RMS_Sph_mat: {}".format(erp_flo_output_dir + "rms_mat_sph.jpg"))
    rms_mat_sph, _ = flow_evaluate.RMSE_mat(flo_gt, flo_eva,  True, flo_mask)
    rms_mat_sph_vis = image_io.visual_data(rms_mat_sph, min_ratio, max_ratio)
    image_io.image_save(rms_mat_sph_vis, erp_flo_output_dir + "rms_mat_sph.jpg")


if __name__ == "__main__":
    erp_flo_gt_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.flo"
    erp_flo_eva_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward_dis.flo"
    erp_flo_mask_filepath = config.TEST_data_root_dir + "replica_360/apartment_0/0001_opticalflow_forward.png"
    erp_flo_output_dir = config.TEST_data_root_dir + "replica_360/apartment_0/"

    test_opticalflow_metric(erp_flo_gt_filepath, erp_flo_eva_filepath, erp_flo_mask_filepath, erp_flo_output_dir)
