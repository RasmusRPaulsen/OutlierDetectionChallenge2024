import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json


def test_pdm_outlier_detection(settings):
    """
    """
    print("Running PCA analysis on test set")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "test/surfaces")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    test_results_json = os.path.join(result_dir, "test_results.json")

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

    # Read first mesh to determine the number of points
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
    for idx in all_scan_ids:
        scan_id = idx.strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface.vtk")
        print(f"Reading {i + 1} / {n_samples} : {surf_name}")

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

    # Find outliers by guessing that 25% of the samples are outliers
    amount_outliers = 0.25
    n_outliers = int(amount_outliers * n_samples)
    n_inliers = n_samples - n_outliers
    print(f"Finding {n_outliers} outliers and {n_inliers} inliers")

    # Predict by the distance from the mean
    distances = np.linalg.norm(components, axis=1)
    plt.plot(distances, '.', label="Samples")
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

    plt.plot(pc_1_all, pc_2_all, '.', label="Samples")
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

    # Create results
    test_results = []
    for i in range(n_samples):
        scan_id = all_scan_ids[i].strip()
        # Remember to cast bools to int for json serialization
        test_results.append({"scan_id": scan_id, "outlier": int(outliers[i]),
                                       "outlier_probability": outlier_probs[i],
                                       "outlier_threshold": normalized_threshold})

    # Write results to JSON file
    with open(test_results_json, 'w') as json_file:
        json.dump(test_results, json_file, indent=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test-pdm-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        test_pdm_outlier_detection(config.settings)
