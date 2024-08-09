import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
import SimpleITK as sitk
import json


def analyze_one_segmentation(segmentation_name):
    # L1 has label 20
    label_id = 20

    # Read the segmentation and turn into a numpy array
    try:
        img = sitk.ReadImage(segmentation_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segmentation_name}")
        return None

    segm_np = sitk.GetArrayFromImage(img)

    # Due to the coordinate conventions in SimpleITK and numpy we need to reorder the image
    # mosty usefull if you need coordinate conversions between the image and the numpy array
    segm_np = segm_np.transpose(2, 1, 0)

    binary_segmentation = segm_np == label_id
    if np.sum(binary_segmentation) == 0:
        print(f"Label {label_id} not found in {segmentation_name}")
        return None

    # Compute the volume of the segmentation
    volume_voxels = np.sum(binary_segmentation)
    # use spacing to compute volume in square millimeters
    volume_mm3 = volume_voxels * np.prod(img.GetSpacing())

    return volume_mm3


def compute_segmentation_statistics(settings):
    """
    Compute a set of simple statistics for the segmentations
    """
    print("Running segmentations analysis")
    data_dir = settings["data_dir"]
    crop_dir = os.path.join(data_dir, "train/crops")
    training_list = settings["data_set"]
    result_dir = settings["result_dir"]

    segmentation_out_dir = os.path.join(result_dir, "segmentation_analysis")
    # Create folders if they don't exist
    Path(segmentation_out_dir).mkdir(parents=True, exist_ok=True)
    segmentation_analysis_out = os.path.join(segmentation_out_dir, f"segmentation_analysis.json")

    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Analyse all samples
    segmentation_volumes = []
    i = 0
    for idx in all_scan_ids:
        print(f"Analysing {i + 1} / {len(all_scan_ids)}")
        scan_id = idx.strip()
        segmentation_name = os.path.join(crop_dir, f"{scan_id}_crop_label.nii.gz")
        volume = analyze_one_segmentation(segmentation_name)
        if volume is not None:
            segmentation_volumes.append(volume)
        i += 1

    if len(segmentation_volumes) == 0:
        print(f"No valid segmentations found")
        return

    # Compute volume statistics
    segmentation_volumes = np.array(segmentation_volumes)
    min_volume = np.min(segmentation_volumes)
    max_volume = np.max(segmentation_volumes)
    mean_volume = np.mean(segmentation_volumes)
    std_volume = np.std(segmentation_volumes)
    print(f"Volume statistics: min {min_volume}, max {max_volume}, mean {mean_volume}, std {std_volume}")

    # Save the results as a JSON file
    segmentation_results = {"min_volume": min_volume, "max_volume": max_volume, "mean_volume": mean_volume,
                            "std_volume": std_volume}
    with open(segmentation_analysis_out, 'w') as json_file:
        json.dump(segmentation_results, json_file, indent=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-segmentation_outlier_detection')
    config = DTUConfig(args)
    if config.settings is not None:
        compute_segmentation_statistics(config.settings)


