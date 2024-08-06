import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import random


def create_custom_data_splits(settings):
    print(f"Creating custom data splits")
    train_size = 100
    validation_size = 100
    # How many of the validation samples are outliers
    outlier_size = 50

    data_dir = settings["data_dir"]
    training_list = settings["data_set"]
    train_out = os.path.join(data_dir, f"custom_train_list_{train_size}.txt")
    validation_out = os.path.join(data_dir, f"custom_validation_list_{validation_size}.txt")

    training_id_list_file = os.path.join(data_dir, training_list)
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
        create_custom_data_splits(config.settings)

