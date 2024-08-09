import vtk
import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


def save_synthesized_shape(input_shape, point_values, out_name):
    pd_new = vtk.vtkPolyData()
    pd_new.DeepCopy(input_shape)

    n_points = pd_new.GetNumberOfPoints()
    if n_points * 3 != len(point_values):
        print(f"Number of points in input shape x3 {3 * n_points} and number of values is {len(point_values)}")
        return

    for i in range(n_points):
        p_new = [point_values[i*3], point_values[i*3+1], point_values[i*3+2]]
        pd_new.GetPoints().SetPoint(i, p_new)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_name)
    writer.SetInputData(pd_new)
    writer.Write()


def vtk_to_vector(pd):
    """
    Takes as input a VTK polydata object and returns a numpy array of shape (n_points * 3, 1)
    with elements (x_0, y_0, z_0,..., x_(n-1), y_(n-1), z_(n-1))
    """
    n_points = pd.GetNumberOfPoints()
    vec = np.zeros(n_points * 3)
    for i in range(n_points):
        p = pd.GetPoint(i)
        vec[i*3] = p[0]
        vec[i*3+1] = p[1]
        vec[i*3+2] = p[2]

    return vec


def compute_pca_analysis(settings):
    """
    Do a classical PCA based shape analysis.
    - We assume that the shapes are already aligned (no need for Procrustes)
    - Find the mean shape
    - Then do a PCA analysis to find the major modes of variation
    - Apply the major modes of variation to the mean shape and write the resulting shapes to file
    """
    print("Running PCA analysis")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "train/surfaces")
    training_list = settings["data_set"]
    result_dir = settings["result_dir"]

    pca_dir = os.path.join(result_dir, "pca_analysis")

    # Create folders if they don't exist
    Path(pca_dir).mkdir(parents=True, exist_ok=True)
    pca_analysis_out = os.path.join(pca_dir, f"pca_analysis.pkl")
    mean_shape_name = os.path.join(pca_dir, f"mean_shape.vtk")

    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Read first mesh to determine the number of points
    # we also keep it for later use - to synthesize shapes
    id_0 = all_scan_ids[0].strip()
    surf_name = os.path.join(surface_dir, f"{id_0}_surface.vtk")
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
        print(f"Reading {i + 1} / {n_samples}")
        scan_id = idx.strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface.vtk")
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
            data_matrix[i, j*3] = p[0]
            data_matrix[i, j*3+1] = p[1]
            data_matrix[i, j*3+2] = p[2]
        i += 1

    average_shape = np.mean(data_matrix, 0)
    save_synthesized_shape(first_surface, average_shape, mean_shape_name)

    n_components = 10
    print(f"Computing PCA with {n_components} components")
    shape_pca = PCA(n_components=n_components)
    shape_pca.fit(data_matrix)
    components = shape_pca.transform(data_matrix)

    # https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
    print(f"Saving {pca_analysis_out}")
    with open(pca_analysis_out, 'wb') as pickle_file:
        pickle.dump(shape_pca, pickle_file)

    plt.plot(shape_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    pc_1 = components[:, 0]
    pc_2 = components[:, 1]

    extreme_pc_1_shape_m = np.argmin(pc_1)
    extreme_pc_1_shape_p = np.argmax(pc_1)
    extreme_pc_2_shape_m = np.argmin(pc_2)
    extreme_pc_2_shape_p = np.argmax(pc_2)

    print(f'PC 1 extreme minus shape: {all_scan_ids[extreme_pc_1_shape_m]}')
    print(f'PC 1 extreme plus shape: {all_scan_ids[extreme_pc_1_shape_p]}')
    print(f'PC 2 extreme minus shape: {all_scan_ids[extreme_pc_2_shape_m]}')
    print(f'PC 2 extreme plus shape: {all_scan_ids[extreme_pc_2_shape_p]}')

    plt.plot(pc_1, pc_2, '.', label="All shapes")
    plt.plot(pc_1[extreme_pc_1_shape_m], pc_2[extreme_pc_1_shape_m], "*", color="green", label="Extreme shape 1 minus")
    plt.plot(pc_1[extreme_pc_1_shape_p], pc_2[extreme_pc_1_shape_p], "+", color="green", label="Extreme shape 1 plus")
    plt.plot(pc_1[extreme_pc_2_shape_m], pc_2[extreme_pc_2_shape_m], "*", color="red", label="Extreme shape 2 minus")
    plt.plot(pc_1[extreme_pc_2_shape_p], pc_2[extreme_pc_2_shape_p], "+", color="red", label="Extreme shape 2 plus")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Shapes in PCA space")
    plt.legend()
    plt.show()

    n_modes = 5
    print(f"Synthesizing shapes using {n_modes} modes")
    for m in range(n_modes):
        synth_shape_plus = average_shape + 3 * np.sqrt(shape_pca.explained_variance_[m]) * shape_pca.components_[m, :]
        synth_shape_minus = average_shape - 3 * np.sqrt(shape_pca.explained_variance_[m]) * shape_pca.components_[m, :]
        pca_plus_out = os.path.join(pca_dir, f"shape_pca_mode_{m}_plus.vtk")
        save_synthesized_shape(first_surface, synth_shape_plus, pca_plus_out)
        pca_minus_out = os.path.join(pca_dir, f"shape_pca_mode_{m}_minus.vtk")
        save_synthesized_shape(first_surface, synth_shape_minus, pca_minus_out)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-pdm_outlier_detection')
    config = DTUConfig(args)
    if config.settings is not None:
        compute_pca_analysis(config.settings)


