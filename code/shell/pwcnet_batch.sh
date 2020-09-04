#!/bin/bash


input_folder="/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/replica_seq_data/"
output_folder="/mnt/sda1/workdata/opticalflow_data/replica_360/office_0/pwcnet/"

pwc_net_dir="/mnt/sda1/workspace_linux/PWC-Net/PyTorch/"
pwc_net_script=${pwc_net_dir}script_pwc.py

cd ${pwc_net_dir}
source /mnt/sda1/workenv_linux/python_2_7_pytorch/bin/activate

of_fw_file_name_list=()
of_bw_file_name_list=()
rgb_file_name_list=()
for i in `seq -f "%04g" 0 17`
do
  rgb_file_name_list[10#$i]=$input_folder${i}_rgb.jpg
  of_bw_file_name_list[10#$i]=$output_folder${i}_opticalflow_backward.flo
  of_fw_file_name_list[10#$i]=$output_folder${i}_opticalflow_forward.flo
done

# compute forward optical flow
for ((i = 0; i < ${#rgb_file_name_list[@]}; ++i)); 
do
    current_index=$i
    next_index=$((($i + 1 + ${#rgb_file_name_list[@]}) % ${#rgb_file_name_list[@]}))

    of_file_name=${of_fw_file_name_list[$current_index]}

    current_image=${rgb_file_name_list[$current_index]}
    next_image=${rgb_file_name_list[$next_index]}

    echo "-------- $i -------"
    echo "comput forward optical flow: $of_file_name"
    echo "from image $current_image"
    echo "to $next_image"
    #echo "python $pwc_net_script $current_image $next_image $of_file_name"
    python $pwc_net_script $current_image $next_image $of_file_name
done

# compute backward optical flow
for ((i = 0; i < ${#rgb_file_name_list[@]}; ++i)); 
do
    previous_index=$((($i - 1 + ${#rgb_file_name_list[@]}) % ${#rgb_file_name_list[@]}))
    current_index=$i

    of_file_name=${of_bw_file_name_list[$current_index]}

    current_image=${rgb_file_name_list[$current_index]}
    previous_image=${rgb_file_name_list[$previous_index]}

    echo "-------- $i -------"
    echo "comput forward optical flow: $of_file_name"
    echo "from image $current_image"
    echo "to $previous_image"
    # echo "python $pwc_net_script $current_image $previous_image $of_file_name"
    python $pwc_net_script $current_image $previous_image $of_file_name
done

