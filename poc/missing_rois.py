import nilearn as nl
import nilearn.image as img
import nilearn.plotting as nplt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from multiprocessing import Pool

data_loc = "/home/012/b/be/dal144042/fmri_data"
participants_df = pd.read_csv(f"{data_loc}/participants_200_list.csv")[["subid", "image", "run"]].drop_duplicates()
mni_template = f"{data_loc}/mni/MNI152_T1_2mm_brain.nii.gz"
v85_fmri = f"{data_loc}/perfect_example/sub-NDARINV003RTV85_ses-baselineYear1Arm1_task-sst_run-02LN.feat"
v85_ln = f"{v85_fmri}/sub-NDARINV003RTV85_filtered_func_data_LN.nii.gz"

def get_binary_region(mat):
    mat = mat
    binary_mat = np.zeros(mat.shape)
    binary_mat[(mat != 0)] = 1
    return binary_mat.astype(np.uint8)

template = img.load_img(mni_template)
template_mat = template.get_fdata()
template_mat = get_binary_region(template_mat)

def get_diff_area(image):
    mat = image.get_fdata()
    mat = get_binary_region(mat)
    xored = np.bitwise_xor(mat, template_mat)
    area = sum(xored.ravel().astype(np.float64))
    return area

def verify(area):
    return int(area < 80000)

def data_path(subid, image, run):
    return f"{data_loc}/sub-{subid}_ses-baselineYear1Arm1_task-sst_{run}LN.feat/stats_roi/sub-{subid}_{image}_{run}_LN.nii.gz"

def get_result(fmri):
    fmri_path = data_path(fmri[0], fmri[1], fmri[2])
    if os.path.exists(fmri_path):
        image = None
        if len(img.load_img(fmri_path).shape) == 4:
            image = img.load_img(img.index_img(fmri_path, 0))
        else:
            image = img.load_img(fmri_path)
        area = get_diff_area(image)
        passed = verify(area)
        return (*fmri, area, passed)
    return (*fmri, -1, -1)

def main():
    if len(sys.argv) == 1:
        fmris = list(participants_df.itertuples(index = False, name = None))
        results = []
        with Pool() as pool:
            results = pool.map(get_result, fmris)
        num_passed = 0
        total = 0
        for result in results:
            passed = result[-1]
            if passed != -1:
                num_passed += passed
                total += 1
        with open("./good_scores.txt", "w", encoding = "utf-8") as good_scores_file, \
            open("./bad_scores.txt", "w", encoding = "utf-8") as bad_scores_file, \
            open("./scores.txt", "w", encoding = "utf-8") as scores_file:
            print("subid", "image", "run", "score", "passed", f"({num_passed}/{total})", file = good_scores_file)
            print("subid", "image", "run", "score", "passed", f"({num_passed}/{total})", file = bad_scores_file)
            print("subid", "image", "run", "score", "passed", f"({num_passed}/{total})", file = scores_file)
            for result in results:
                print(*result, file = scores_file)
                if result[-1] == 1:
                    print(*result, file = good_scores_file)
                elif result[-1] == 0:
                    print(*result, file = bad_scores_file)
    elif len(sys.argv) == 4:
        subid = sys.argv[1]
        image = sys.argv[2]
        run = sys.argv[3]
        fmri_path = data_path(subid, image, run)
        im = None
        if len(img.load_img(fmri_path).shape) == 4:
            im = img.load_img(img.index_img(fmri_path, 0))
        else:
            im = img.load_img(fmri_path)
        area = get_diff_area(im)
        nplt.plot_glass_brain(im, display_mode = "x", title = f"{area}", output_file = "./plot.png")

main()
