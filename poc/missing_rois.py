import nilearn as nl
import nilearn.image as img
import nilearn.plotting as nplt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

data_loc = "/home/012/b/be/dal144042/fmri_data"
participants_df = pd.read_csv(f"{data_loc}/participants_200_list.csv")
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

def verify(image):
    return get_diff_area(image) < 80000

def data_path(subid, image, run):
    return f"{data_loc}/sub-{subid}_ses-baselineYear1Arm1_task-sst_{run}LN.feat/stats_roi/sub-{subid}_{image}_{run}_LN.nii.gz"

def main():
    if len(sys.argv) == 1:
        fmris = list(participants_df[["subid", "image", "run"]].itertuples(index = False, name = None))
        images = []
        for fmri in fmris:
            fmri_path = data_path(fmri[0], fmri[1], fmri[2])
            if len(img.load_img(fmri_path).shape) == 4:
                images.append(img.load_img(img.index_img(fmri_path, 0)))
            else:
                images.append(img.load_img(fmri_path))
        areas = [get_diff_area(image) for image in images]
        checks = [verify(image) for image in images]
        # Reference: https://nilearn.github.io/dev/auto_examples/01_plotting/plot_transparency.html#sphx-glr-auto-examples-01-plotting-plot-transparency-py
        # fig, axes = plt.subplots(4, 6, figsize = (15, 15))
        # for i, _ in enumerate(images):
        #     image = images[i]
        #     area = areas[i]
        #     row = int(i / 6)
        #     col = i % 6
        #     nplt.plot_glass_brain(image, display_mode = "x", title = f"{area}", axes = axes[row][col], colorbar = False)
        # fig.savefig("./results/plot.png")
        results = []
        num_passed = 0
        for fmri, area, check in zip(fmris, areas, checks):
            results.append((*fmri, area, check))
            num_passed += int(check)
        with open("./scores.txt", "w", encoding = "utf-8") as file:
            print("subid", "image", "run", "score", "passed", f"({num_passed}/{len(results)})", file = file)
            for result in results:
                print(*result, file = file)
        # good_image = img.load_img(img.index_img(v85_ln, 0))
        # good_area = get_diff_area(good_image)
        # nplt.plot_glass_brain(good_image, display_mode = "x", title = f"{good_area}", colorbar = False, output_file = "./results/good_plot.png")
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