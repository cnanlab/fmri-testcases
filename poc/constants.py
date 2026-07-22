import os
#data_loc = "/mnt/storage/failed_roi/participants_200"
data_loc = "/mnt/WarrenNAS/Projects/processed_mri/sst/baseline"
curr_loc = f"{os.getcwd()}/export"
participants_file = f"{curr_loc}/subject_run_latest_path.csv" #f"{data_loc}/participants_200_list.csv"
mni_template = f"/mnt/storage/masks/MNI152_T1_2mm_brain.nii.gz"
#v85_fmri = f"{data_loc}/perfect_example/sub-NDARINV003RTV85_ses-baselineYear1Arm1_task-sst_run-02LN.feat"
#v85_ln = f"{v85_fmri}/sub-NDARINV003RTV85_filtered_func_data_LN.nii.gz"