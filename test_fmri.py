from collections import namedtuple
import glob
import pytest
import nibabel as nib
import numpy as np
from joblib import delayed, Parallel
from itertools import compress
import os
import logging
import pandas as pd


# Configure logging
logging.basicConfig(
level=logging.DEBUG,
filemode='a',
format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[
logging.FileHandler("app.log"), # Log to file
logging.StreamHandler() # Log to stdout
]
)

log = logging.getLogger(__name__)
log.propagate = True

@pytest.fixture
def config():
    
    # paths = dict(zip(['parent', 'current', 'ev_path', '', ''], 
    #          ['/mnt/storage/processed_mri/sst/', 
    #             'baseline/', 
    #             'stats/'
    #             ]))
    paths = dict(zip(['parent', 'current', 'ev_path', '', ''], 
             ['/mnt/WarrenNAS/Data/processed_mri/SST/', 
                #'2025-12-17_07-55-33/', 
                "2025-12-18_11-02-25/",
                'stats/'
                ]))
    
    config = dict(paths=namedtuple("Paths",paths)(**paths),
                expected_preprocessed_file_shape = (90, 90, 60),
                expected_volume= (437,),
                preprocessed_file = "filtered_func_data",
                ev_prefixes_mins_maxs=[['pe', (1,12)],['zfstats', (1,18)]],
                mri_extn = ".nii.gz"
                )

    return namedtuple("config",config.keys())(**config)

@pytest.fixture
def processed_parti_folders(config):
    files_processed_arr = glob.glob(config.paths.parent+config.paths.current+"*", recursive=True)
    return files_processed_arr

@pytest.fixture
def processed_fmri_files(config):
    files_containing_data_arr = glob.glob(config.paths.parent+config.paths.current+"*/"+config.preprocessed_file+config.mri_extn, recursive=True)
    return files_containing_data_arr

@pytest.fixture
def processed_fmri_folders(config):
    files_containing_data_arr = glob.glob(config.paths.parent+config.paths.current+"*/"+config.preprocessed_file+config.mri_extn, recursive=True)
    fmri_folders = list(map(lambda f:os.path.dirname(f), files_containing_data_arr))
    return fmri_folders

class Test_FileDetails:
    
    def test_processed_participants(self, processed_parti_folders, processed_fmri_files):
        log.info(f"Examining {len(processed_parti_folders)} folders")
        assert len(processed_parti_folders) == len(processed_fmri_files), f"Out of {len(processed_parti_folders)}, {len(processed_parti_folders) - len(processed_fmri_files)} preprocessed filtered_func_data.nii.gz files are missing"

    def test_processed_stats_file_count(self, config, processed_fmri_folders):
        print(f"Examining {len(processed_fmri_folders)} folders")
        missing_arr = sum(list(map(lambda f: len(list(os.walk(f+"/stats/")))<100, processed_fmri_folders)) )
        
        assert sum(missing_arr) == 0, f"Out of {len(processed_fmri_folders)} processed participants, {sum(missing_arr)} do not have all files in 1st level stats"

    def test_processed_stats(self, config, processed_fmri_folders):
        print(f"Examining {len(processed_fmri_folders)} folders")
        missing_arr = list(map(lambda f: not os.path.exists(f + "/" + config.paths.ev_path), processed_fmri_folders))        
        
        assert sum(missing_arr) == 0, f"Out of {len(processed_fmri_folders)} processed participants, {sum(missing_arr)} do not have 1st level stats"

    def test_processed_file_shape(self, config, processed_fmri_files):
        log.info(f"Examining {len(processed_fmri_files)} participants")

        expected_shape = config.expected_preprocessed_file_shape + config.expected_volume

        shape_arr = list(map(lambda file:nib.load(file).shape, processed_fmri_files))
        unmatched_arr = list(map(lambda shape: shape != expected_shape, shape_arr))
    
        unmatched_shape_arr = list(compress(shape_arr, unmatched_arr))

        assert sum(unmatched_arr) == 0, f"Expected {expected_shape}; not found in {sum(unmatched_arr)} files; e.g., {unmatched_shape_arr[0]} "

    def test_evs_present(self, config, processed_fmri_folders):
        log.info(f"Examining {len(processed_fmri_folders)} folders")
        missing_arr = []
        missing_path_eg = ""
        for folder in processed_fmri_folders:
            for prefix, (min_id, max_id) in config.ev_prefixes_mins_maxs:
                for id in range(min_id, max_id):
                    path = folder + "/" + config.paths.ev_path + prefix + str(id) + config.mri_extn
                    missing_arr += [not os.path.exists(path)]
                    if not os.path.exists(path):
                        missing_path_eg = folder
        
        assert sum(missing_arr) == 0, f" Out of {len(missing_arr)}, {sum(missing_arr)} EV folders missing; e.g., {missing_path_eg}"

    def test_runs_present(self, config, processed_fmri_folders):
        log.info(f"Examining {len(processed_fmri_folders)} folders")
        part_ids = []
        #part_run_count = {}
        for folder in processed_fmri_folders:
            p_id = folder.split("sub-")[1].split("_ses")[0]
            run = folder.split("run-")[1].split(".feat")[0]
            part_ids.append(
                {
                    "part_id":p_id,
                    "run":run
                }
            )
        df_ids = pd.DataFrame(part_ids)
        df_missing = df_ids[df_ids.groupby("part_id")["run"].transform("count").values < 2]
        missing_runs = df_missing.shape[0]
        df_missing.sort_values("part_id").to_csv(f"export/missing_runs_{config.paths.current.remove('//','')}.csv", index=None)
        assert missing_runs == 0, f"{missing_runs} Missing"

#if __name__=="__main__":
#    print (glob.glob("/mnt/storage/processed_mri/sst/baseline/*/filtered_func_data.nii.gz", recursive=True))

class Test_ImageQuality:
    def test_processed_image_missing_voxels(self, config, processed_fmri_files):
        log.info(f"Examining {len(processed_fmri_files)} participants")

        #expected_shape = config.expected_preprocessed_file_shape + config.expected_volume

        def fn (f_t):
            try:
                file_np = nib.load(f_t).get_fdata()
                f_c = np.count_nonzero(file_np)
                f_n = file_np.size
                f_p = 1-(f_c / f_n)
            except:
                f_c = None
                f_n = None
                f_p = None
            finally:
                dict_t = dict(
                    file_name = f_t,
                    nonzero_voxel = f_c,
                    total_voxel = f_n,
                    perc_voxel_missing = f_p
                )
            return dict_t
        df_rez = pd.DataFrame.from_dict(Parallel(n_jobs=2)(delayed(fn)(f_t) for f_t in processed_fmri_files))
        df_rez.to_csv(f"export/evs_missing_information_{config.paths.current.remove('//','')}.csv")
        failure = np.where(df_rez["nonzero_voxel"] / df_rez["total_voxel"] > 0.20, False, True)
        failure = failure.sum()
        assert failure == 0, f"{failure} failed! check exported csv for missing EV voxels"