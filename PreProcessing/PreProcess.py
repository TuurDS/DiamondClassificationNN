import numpy as np
import os
import shutil
import random
import time
from stl import mesh
import csv
from scipy.ndimage.morphology import binary_erosion

from util.utilFunctions import batch, get_dir_size, FindMaxShape
from util.exportData import exportData
from PreProcessing import slice

def getTrainAndTestData(models, amount, testAmount, baseInput):
    train_file_paths = []
    for name in models:
        for i in range(1, amount + 1):
            train_file_paths.append(baseInput + name + "-" + str(i) + ".stl")


    # now do the same for the test data
    test_file_paths = []
    for name in models:
        for i in range(amount + 1, amount + testAmount + 1):
            test_file_paths.append(baseInput + name + "-" + str(i) + ".stl")

    return train_file_paths, test_file_paths

def convert_meshes(meshes, resolution=64, voxel_size=None, parallel=True):

    scale, shift, shape = slice.calculate_scale_shift(meshes, resolution-1, voxel_size)
    vol = np.zeros(shape[::-1], dtype=np.int8)

    for mesh_ind, org_mesh in enumerate(meshes):
        slice.scale_and_shift_mesh(org_mesh, scale, shift)
        cur_vol = slice.mesh_to_plane(org_mesh, shape, parallel)
        vol[cur_vol] = mesh_ind + 1
    return vol, scale, shift

def convertToMesh(input_file_paths):
    meshes = []
    for input_file_path in input_file_paths:
        # Load mesh object
        mesh_obj = mesh.Mesh.from_file(input_file_path)

        #region auto rotate + rotate from metadata (not used/working)
        # Calculate surface areas of all sides
        # surface_areas = np.sum(np.abs(mesh_obj.normals), axis=1)

        # # Find the side with the largest surface area
        # largest_side_idx = np.argmax(surface_areas)

        # # Calculate normal vector of the largest side
        # largest_side_normal = mesh_obj.normals[largest_side_idx]

        # # Calculate rotation angle to align the normal vector with the z-axis
        # angle = np.arccos(np.dot(largest_side_normal, [0,0,1]))

        # # Calculate rotation axis
        # rotation_axis = np.cross(largest_side_normal, [0,0,1])

        # # Rotate mesh
        # mesh_obj.rotate(rotation_axis, angle)

        # # Rotate mesh
        # file_name = input_file_path.split("/")[-1]

        # with open(r'C:/Users/Tuur/projects/bachlorproef/pyton/data/metadata.csv', 'r') as csvfile:
        #     csvreader = csv.reader(csvfile, delimiter=';')
        #     # Loop through rows
        #     for row in csvreader:
        #         # Check if search value is in last element of row
        #         if file_name in row[-1]:
        #             # Extract rotation angles from second to fourth element of row
        #             rx = -float(row[1])
        #             ry = -float(row[2]) # + 180.0  # apply 180 degree shift to y rotation
        #             rz = -float(row[3])
                    
        #             # Define rotation matrix
        #             r_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        #             r_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        #             r_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        #             rotation_matrix = np.dot(r_x, np.dot(r_y, r_z))
                    
        #             # Apply rotation
        #             mesh_obj.rotate_using_matrix(rotation_matrix)
        #             break
        #endregion

        #region random 180 degree rotation
        if random.random() < 0.5:
            mesh_obj.rotate([0,0,1], np.deg2rad(0))
        else:
            mesh_obj.rotate([0,0,1], np.deg2rad(180))
        #endregion

        #random Z rotation between -180 and 180 degrees
        # mesh_obj.rotate([0,0,1], np.deg2rad(random.randint(-180, 180)))

        # mesh_obj.save('output.stl')

        # Convert to numpy array
        org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))

        meshes.append(org_mesh)

    return meshes

def load_stl_file(filename, resolution = 64, parallel = True):
    meshes = convertToMesh([filename])
    voxel_data, scale, shift = convert_meshes(meshes, resolution=resolution, parallel=parallel)

    #region Hollow out the shape
    eroded_voxels = binary_erosion(voxel_data, iterations=1)
    voxel_data = np.logical_xor(voxel_data, eroded_voxels).astype(bool)
    #endregion

    # exportData([voxel_data], [0], r"C:/", ["Brilliant"], True, True)
    # print(filename)

    return voxel_data

def voxelize_and_save_batch(batch_paths, batch_idx, name, basePath, models, resolution=64, parallel=True):
    #generate the data and labels
    data = []
    labels = []
    for i, filename in enumerate(batch_paths):
        # Load the voxel grid
        voxel_grid = load_stl_file(filename, resolution)
        print(f'batch {batch_idx} voxelized file {str(i+1)}')

        data.append(voxel_grid)

        for j, model in enumerate(models):
            if model == filename.split("/")[-1].split("-")[0]:
                labels.append(j)
    
    # compute x_max, y_max, z_max
    x_max = max([arr.shape[0] for arr in data])
    x_max = ((x_max + 7) // 8) * 8  # round up to nearest multiple of 8
    y_max = max([arr.shape[1] for arr in data])
    y_max = ((y_max + 7) // 8) * 8  # round up to nearest multiple of 8
    z_max = max([arr.shape[2] for arr in data])
    z_max = ((z_max + 7) // 8) * 8  # round up to nearest multiple of 8

    # pad data if accidentally did not set resolution to multiple of 8 for compatibility with compression
    padded_batch_data = []
    for arr in data:
        x_pad_before = (x_max - arr.shape[0]) // 2
        x_pad_after = x_max - arr.shape[0] - x_pad_before
        y_pad_before = (y_max - arr.shape[1]) // 2
        y_pad_after = y_max - arr.shape[1] - y_pad_before
        z_pad_before = (z_max - arr.shape[2]) // 2
        z_pad_after = z_max - arr.shape[2] - z_pad_before
        padded_arr = np.pad(arr, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)), mode='constant')
        padded_arr = np.packbits(padded_arr, axis=-1)
        padded_batch_data.append(padded_arr)
    
    #ensure that the output path exists
    batch_idx_str = str(batch_idx).zfill(3)
    os.makedirs(os.path.dirname(os.path.join(basePath,f"{name}_batch_{batch_idx_str}_data.npy")), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(basePath,f"{name}_batch_{batch_idx_str}_labels.npy")), exist_ok=True)

    np.save(os.path.join(basePath,f"{name}_batch_{batch_idx_str}_data.npy"), np.asarray(padded_batch_data))
    np.save(os.path.join(basePath,f"{name}_batch_{batch_idx_str}_labels.npy"), np.asarray(labels))

def preprocess_and_save_batches(batchPath, x_max, y_max, z_max, start_time):
    # Find all data files
    data_files = [f for f in os.listdir(batchPath) if "data" in f]

    for i, batch_file in enumerate(data_files):
        print(f'preprocessing batch file {i+1} :')
        # Load batch data and decompress
        batch_data = np.unpackbits(np.load(os.path.join(batchPath, batch_file)), axis=-1).astype(bool)

        # Initialize empty array to store padded voxel grids
        padded_batch_data = []

        # Loop through each voxel grid in batch
        for i, voxel_grid in enumerate(batch_data):
            print(f'padding file {str(i+1)}')
            # Get dimensions of voxel grid
            x, y, z = voxel_grid.shape

            # Compute padding values for each dimension
            x_pad_before = (x_max - x) // 2
            x_pad_after = x_max - x - x_pad_before
            y_pad_before = (y_max - y) // 2
            y_pad_after = y_max - y - y_pad_before
            z_pad_before = (z_max - z) // 2
            z_pad_after = z_max - z - z_pad_before

            # Pad the array equally from left, right, top, bottom and side to side
            padded_arr = np.pad(voxel_grid, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)), mode='constant')
            padded_arr = np.packbits(padded_arr, axis=-1)
            padded_batch_data.append(padded_arr)

        print('converting to numpy array...')
        # Convert list of padded voxel grids to numpy array
        padded_batch_data = np.asarray(padded_batch_data)
        print(f'saving... {time.time() - start_time}')
        # Save padded batch data back to file
        np.save(os.path.join(batchPath, batch_file), padded_batch_data)


def PreProcess(models, amount, testAmount, exportPath, baseInput, batchPath, batchSize, resolution, export_vox):
    start_time = time.time()

    train_file_paths, test_file_paths = getTrainAndTestData(models, amount, testAmount, baseInput)

    # shuffle the train_file_paths and test_file_paths
    random.shuffle(train_file_paths)
    random.shuffle(test_file_paths)

    # batch the train_file_paths and test_file_paths
    train_batches = list(batch(train_file_paths, batchSize))
    test_batches = list(batch(test_file_paths, batchSize)) 

    print("voxelizing")
    if os.path.exists(batchPath):
        shutil.rmtree(batchPath)

    for i, batchFiles in enumerate(train_batches):
        print(f"voxelizing train batch {str(i+1)} {time.time() - start_time}")
        voxelize_and_save_batch(batchFiles, i+1, "train", batchPath, models, resolution=resolution)

    for i, batchFiles in enumerate(test_batches):
        print(f"voxelizing test batch {str(i+1)} {time.time() - start_time}")
        voxelize_and_save_batch(batchFiles, i+1, "test", batchPath, models, resolution=resolution)

    print("done with voxelizing")
    print(f"Elapsed time: {time.time() - start_time} seconds")
    print(f"Total size: {get_dir_size(batchPath) / 1000000} MB")


    # if shape is not equal for all models this will normalize the shape by padding the models with zeros (dimensions will be a multiple of 8 for compressing)
    print("preprocessing...")
    print("finding shape...")
    x_max, y_max, z_max = FindMaxShape(batchPath)
    print("shape found: " + str(x_max) + " " + str(y_max) + " " + str(z_max))
    preprocess_and_save_batches(batchPath, x_max, y_max, z_max, start_time)
    print("done with preprocessing")


    print(f"Elapsed time: {time.time() - start_time} seconds")
    print(f"Total size: {get_dir_size(batchPath) / 1000000} MB")


    # export a batch to visualize via png's and export vox files to test manually
    if(resolution <= 128 and export_vox):    
        data_files = [f for f in os.listdir(batchPath) if "data" in f]
        data_file = data_files[random.randint(0, len(data_files) - 1)]
        packedBatchData = np.load(batchPath + "/" + data_file)
        BatchLabels = np.load(batchPath + "/" + data_file.replace("data", "labels"))
        #loop over packedBatchData and unpack each element
        unpackedBatchData = []
        for i in range(len(packedBatchData)):
            unpackedBatchData.append(np.unpackbits(packedBatchData[i], axis=-1).astype(bool))
        exportData(np.asarray(unpackedBatchData),BatchLabels,exportPath,models,True,True)


def PreProcessSingleFile(filepath,batchPath, resolution = 64):

    voxel_grid = load_stl_file(filepath, resolution)
    x_max, y_max, z_max = FindMaxShape(batchPath)
    x, y, z = voxel_grid.shape
    
    # Compute padding values for each dimension
    x_pad_before = (x_max - x) // 2
    x_pad_after = x_max - x - x_pad_before
    y_pad_before = (y_max - y) // 2
    y_pad_after = y_max - y - y_pad_before
    z_pad_before = (z_max - z) // 2
    z_pad_after = z_max - z - z_pad_before

    # Pad the array equally from left, right, top, bottom and side to side
    padded_arr = np.pad(voxel_grid, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)), mode='constant')
    return padded_arr