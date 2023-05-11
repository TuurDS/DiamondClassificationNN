import numpy as np
import trimesh
from pyvox.models import Vox
from pyvox.writer import VoxWriter

"""testing voxelization of stl files"""
def load_stl_file(filename, resolution = 100, parallel = True):
    # meshes = convertToMesh([filename])
    # voxel_data, scale, shift = convert_meshes(meshes, resolution=voxel_size, parallel=parallel)

    mesh = trimesh.load(filename)
    # Determine the pitch of each voxel based on the specified dimensions
    pitch = np.array(mesh.bounding_box.volume) / np.array((resolution, resolution, resolution))
    # Voxelize the mesh using the specified pitch
    volume = mesh.voxelized(pitch=pitch)
    # Convert the voxel grid to a boolean numpy array
    voxel_data = volume.matrix.astype(bool)

    return voxel_data

