import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json


def validate_pdm_outlier_detection(settings):
    """
    """
    print("Running PCA analysis on validation set")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "train/surfaces")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    validation_results_json = os.path.join(result_dir, "validation_results.json")

    pca_dir = os.path.join(result_dir, "pca_analysis")

    pca_analysis_in = os.path.join(pca_dir, f"pca_analysis.pkl")
    mean_shape_name = os.path.join(pca_dir, f"mean_shape.vtk")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    print(f"Loading PCA from {pca_analysis_in}")
    with open(pca_analysis_in, 'rb') as picklefile:
        shape_pca = pickle.load(picklefile)

    # Read the mean mesh to determine the number of points
    # we also keep it for later use - to synthesize shapes
    # id_0 = all_scan_ids[0].strip()
    # surf_name = os.path.join(surface_dir, f"{id_0}_surface.vtk")
    surf_name = mean_shape_name
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surf_name)
    reader.Update()
    first_surface = reader.GetOutput()

    n_points = first_surface.GetNumberOfPoints()
    # Three features per point (x, y, z)
    n_features = n_points * 3
    n_samples = len(all_scan_ids)
    print(f"Creating date matrix of size {n_samples} x {n_features}")
    data_matrix = np.zeros((n_samples, n_features))

    # Now read all meshes
    i = 0
    outliers_gt = np.zeros(n_samples, dtype=bool)
    for idx in all_scan_ids:
        scan_id = idx[0].strip()

        outlier_type = idx[1].strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface{outlier_type}.vtk")
        print(f"Reading {i + 1} / {n_samples} : {surf_name}")
        if outlier_type != "":
            outliers_gt[i] = True

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(surf_name)
        reader.Update()
        surface = reader.GetOutput()
        n_p = surface.GetNumberOfPoints()
        if n_p != n_points:
            print(f"Number of points in {scan_id} is {n_p} and it should be {n_points}")
            return
        for j in range(n_p):
            p = surface.GetPoint(j)
            data_matrix[i, j * 3] = p[0]
            data_matrix[i, j * 3 + 1] = p[1]
            data_matrix[i, j * 3 + 2] = p[2]
        i += 1

    # Turns out we should NOT subtract the mean before pca transform
    # average_shape = vtk_to_vector(first_surface)
    # data_matrix = data_matrix - average_shape
    components = shape_pca.transform(data_matrix)

    pc_1_all = components[:, 0]
    pc_2_all = components[:, 1]

    extreme_pc_1_shape_m = np.argmin(pc_1_all)
    extreme_pc_1_shape_p = np.argmax(pc_1_all)
    extreme_pc_2_shape_m = np.argmin(pc_2_all)
    extreme_pc_2_shape_p = np.argmax(pc_2_all)

    print(f'PC 1 extreme minus shape: {all_scan_ids[extreme_pc_1_shape_m]}')
    print(f'PC 1 extreme plus shape: {all_scan_ids[extreme_pc_1_shape_p]}')
    print(f'PC 2 extreme minus shape: {all_scan_ids[extreme_pc_2_shape_m]}')
    print(f'PC 2 extreme plus shape: {all_scan_ids[extreme_pc_2_shape_p]}')

    pc_1 = components[~outliers_gt, 0]
    pc_2 = components[~outliers_gt, 1]

    pc_1_out = components[outliers_gt, 0]
    pc_2_out = components[outliers_gt, 1]

    plt.plot(pc_1, pc_2, '.', label="Normals")
    plt.plot(pc_1_out, pc_2_out, '*', label="Outliers")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Shapes in PCA space")
    plt.legend()
    plt.show()

    # Find outliers by guessing that 25% of the samples are outliers
    amount_outliers = 0.25
    n_outliers = int(amount_outliers * n_samples)
    n_inliers = n_samples - n_outliers
    print(f"Finding {n_outliers} outliers and {n_inliers} inliers")

    # We classify by the distance from the mean
    distances = np.linalg.norm(components, axis=1)
    plt.plot(distances[~outliers_gt], '.', label="Normal")
    plt.plot(distances[outliers_gt], '*', label="Outliers")
    plt.xlabel('Sample')
    plt.ylabel('Distance from mean')
    plt.title("Distance from mean in PCA space")
    plt.legend()
    plt.show()

    # sort distances and select the 25% most distant
    sorted_distances = np.sort(distances)
    threshold = sorted_distances[n_inliers]
    print(f"Threshold is {threshold:.1f}")
    outliers = distances >= threshold
    print(f"Found {np.sum(outliers)} outliers")
    print(f"Outliers: {all_scan_ids[outliers]}")
    print(f"Non-outliers: {all_scan_ids[~outliers]}")
    print(f"Outlier distances: {distances[outliers]}")
    print(f"Non-outlier distances: {distances[~outliers]}")
    print(f"Outlier distances mean: {np.mean(distances[outliers])}")
    print(f"Non-outlier distances mean: {np.mean(distances[~outliers])}")
    print(f"Outlier distances std: {np.std(distances[outliers])}")
    print(f"Non-outlier distances std: {np.std(distances[~outliers])}")
    print(f"Outlier distances min: {np.min(distances[outliers])}")
    print(f"Non-outlier distances min: {np.min(distances[~outliers])}")
    print(f"Outlier distances max: {np.max(distances[outliers])}")
    print(f"Non-outlier distances max: {np.max(distances[~outliers])}")

    pc_1_out_pred = components[outliers, 0]
    pc_2_out_pred = components[outliers, 1]

    plt.plot(pc_1, pc_2, '.', label="Normals")
    plt.plot(pc_1_out, pc_2_out, '*', label="GT Outliers")
    plt.plot(pc_1_out_pred, pc_2_out_pred, '+', label="Pred. Outliers")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Shapes in PCA space")
    plt.legend()
    plt.show()

    max_dist = np.max(distances)
    min_dist = np.min(distances)
    outlier_probs = (distances - min_dist) / (max_dist - min_dist)
    normalized_threshold = (threshold - min_dist) / (max_dist - min_dist)

    # Create classification results
    validation_results = []
    for i in range(n_samples):
        scan_id = all_scan_ids[i][0].strip()
        # Remember to cast bools to int for json serialization
        validation_results.append({"scan_id": scan_id, "outlier": int(outliers[i]),
                                       "outlier_probability": outlier_probs[i],
                                       "outlier_threshold": normalized_threshold})

    # Write classification results to file
    with open(validation_results_json, 'w') as json_file:
        json.dump(validation_results, json_file, indent=4)


def compute_outlier_detection_metrics(settings):
    """
    """
    print("Computing outlier detection metrics")
    data_dir = settings["data_dir"]
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    validation_results_json = os.path.join(result_dir, "validation_results.json")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    n_samples = len(all_scan_ids)
    outliers_gt = np.zeros(n_samples, dtype=bool)
    outlier_pred = np.zeros(n_samples, dtype=bool)

    with open(validation_results_json, 'r') as json_file:
        validation_results = json.load(json_file)

    i = 0
    n_predicted_outliers = 0
    for idx in all_scan_ids:
        scan_id = idx[0].strip()
        outlier_type = idx[1].strip()
        if outlier_type != "":
            outliers_gt[i] = True

        for res in validation_results:
            if res["scan_id"] == scan_id:
                outlier_pred[i] = res["outlier"]
                n_predicted_outliers += 1
                break
        i += 1
    print(f"Found {n_predicted_outliers} predicted outliers out of {n_samples} samples")

    # Compute metrics
    tp = np.sum(outliers_gt & outlier_pred)
    tn = np.sum(~outliers_gt & ~outlier_pred)
    fp = np.sum(~outliers_gt & outlier_pred)
    fn = np.sum(outliers_gt & ~outlier_pred)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} N_pred: {n_predicted_outliers} N_samples: {n_samples}")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / n_samples
    cohens_kappa = 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}, "
          f"Cohens kappa: {cohens_kappa:.2f}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='validate-pdm-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        validate_pdm_outlier_detection(config.settings)
        compute_outlier_detection_metrics(config.settings)

