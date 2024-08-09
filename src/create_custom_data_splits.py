import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import random
from pathlib import Path
from shutil import copyfile


def copy_file_lists(settings):
    """
    Copy the file lists from the data directory to the result directory
    """
    data_dir = settings["data_dir"]
    result_dir = settings["result_dir"]

    print(f"Creating output directory {result_dir} and copying file lists from {data_dir}")
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    in_files = ["train_files.txt", "test_files.txt", "test_files_200.txt"]

    for f in in_files:
        in_file = os.path.join(data_dir, f)
        out_file = os.path.join(result_dir, f)
        print(f"Copying {in_file} to {out_file}")
        copyfile(in_file, out_file)


def create_custom_data_splits(settings):
    train_size = 100
    validation_size = 100
    # How many of the validation samples are outliers
    outlier_size = 50

    result_dir = settings["result_dir"]
    training_list = settings["data_set"]
    train_out = os.path.join(result_dir, f"custom_train_list_{train_size}.txt")
    validation_out = os.path.join(result_dir, f"custom_validation_list_{validation_size}.txt")
    print(f"Creating custom data splits:\n{train_out}\n{validation_out}")

    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    if len(all_scan_ids) < train_size + validation_size:
        print(f"Too few samples ({len(all_scan_ids)}) to create custom data splits with {train_size} "
              f"training samples and {validation_size} validation samples")
        return

    print(f"Creating custom data splits with {train_size} training samples and {validation_size} validation samples,"
          f" where {outlier_size} validation samples are outliers")

    # Shuffle the list and select the training and validation samples
    random.shuffle(all_scan_ids)
    train_samples = all_scan_ids[:train_size]
    validation_samples = all_scan_ids[train_size:train_size + validation_size]

    f = open(train_out, "w")
    for samp in train_samples:
        f.write(f"{samp}\n")

    outlier_types = ["_sphere_outlier_mean_std_inpaint", "_sphere_outlier_water", "_warp_outlier"]

    f = open(validation_out, "w")
    idx = 0
    for samp in validation_samples:
        if idx < outlier_size:
            ot = random.randint(0, len(outlier_types) - 1)
            f.write(f"{samp},{outlier_types[ot]}\n")
        else:
            f.write(f"{samp},\n")
        idx += 1


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='create-custom-data-splits')
    config = DTUConfig(args)
    if config.settings is not None:
        copy_file_lists(config.settings)
        create_custom_data_splits(config.settings)

