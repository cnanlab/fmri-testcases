import nilearn as nl
import nilearn.image as img
import nilearn.plotting as nplt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from multiprocessing import Pool
import constants as cs

data_loc = cs.data_loc
participants_df = pd.read_csv(cs.participants_file)[["subid", "image", "run"]].drop_duplicates()
mni_template = cs.mni_template
v85_fmri = cs.v85_fmri
v85_ln = cs.v85_ln

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
    return int(area < 50000)

def data_path(subid, image, run):
    return f"{data_loc}/sub-{subid}_ses-baselineYear1Arm1_task-sst_{run}LN.feat/stats_roi/sub-{subid}_{image}_{run}_LN.nii.gz"

def get_result(fmri):
    fmri_path = data_path(fmri[0], fmri[1], fmri[2])
    if os.path.exists(fmri_path):
        area = 0
        if len(img.load_img(fmri_path).shape) == 4:
            for image in img.iter_img(fmri_path):
                area += get_diff_area(image)
            area /= img.load_img(fmri_path).shape[3]
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
    elif len(sys.argv) > 3:
        subid = sys.argv[1]
        image = sys.argv[2]
        run = sys.argv[3]
        plot_name = "plot"
        if len(sys.argv) > 4:
            plot_name = sys.argv[4]
        fmri_path = data_path(subid, image, run)
        im = None
        if len(img.load_img(fmri_path).shape) == 4:
            im = img.load_img(img.mean_img(fmri_path))
        else:
            im = img.load_img(fmri_path)
        area = get_diff_area(im)
        nplt.plot_glass_brain(im, display_mode = "x", title = f"{area}", output_file = f"./{plot_name}.png")

if __name__ == '__main__':
    main()
