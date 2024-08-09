import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
import SimpleITK as sitk
import json
import matplotlib.pyplot as plt


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


def test_segmentation_outlier_detection(settings):
    """
    """
    print("Running segmentation analysis on test set")
    data_dir = settings["data_dir"]
    crop_dir = os.path.join(data_dir, "test/crops")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    test_results_json = os.path.join(result_dir, "test_results.json")

    segmentation_out_dir = os.path.join(result_dir, "segmentation_analysis")
    segmentation_analysis_in = os.path.join(segmentation_out_dir, f"segmentation_analysis.json")

    # Read the segmentation analysis as a json file
    with open(segmentation_analysis_in, 'r') as json_file:
        segmentation_analysis = json.load(json_file)

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Volume thresholds based on standard deviation
    mean_volume = segmentation_analysis["mean_volume"]
    std_volume = segmentation_analysis["std_volume"]

    lower_volume_threshold = mean_volume - 2 * std_volume
    upper_volume_threshold = mean_volume + 2 * std_volume
    print(f"Volume thresholds: {lower_volume_threshold} - {upper_volume_threshold}")

    n_samples = len(all_scan_ids)

    # Analyse all samples
    normal_volumes = []
    outlier_volumes = []
    test_results = []
    i = 0
    for idx in all_scan_ids:
        print(f"Analysing {i + 1} / {len(all_scan_ids)}")
        scan_id = idx.strip()
        segmentation_name = os.path.join(crop_dir, f"{scan_id}_crop_label.nii.gz")
        volume = analyze_one_segmentation(segmentation_name)
        outlier = False
        if volume is not None:
            if volume < lower_volume_threshold or volume > upper_volume_threshold:
                outlier = True

        if outlier:
            outlier_volumes.append(volume)
        else:
            normal_volumes.append(volume)

        # Remember to cast bools to int for json serialization
        test_results.append({"scan_id": scan_id, "outlier": int(outlier)})
        i += 1

    # Write results to JSON file
    with open(test_results_json, 'w') as json_file:
        json.dump(test_results, json_file, indent=4)


    plt.plot(normal_volumes, '.', label="Normal")
    plt.plot(outlier_volumes, '*', label="Outliers")
    plt.xlabel('Sample')
    plt.ylabel('Volume')
    plt.title("Segmentation volumes")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test-segmentation-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        test_segmentation_outlier_detection(config.settings)
