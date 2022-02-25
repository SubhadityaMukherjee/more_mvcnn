"""
Generate binary occupancy grid from the ModelNet10 dataset.

Stores the voxelized files from the modelnet10 dataset in .npy
format, ready to be fed to entropy_model.py
"""
import os
import sys
import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
import concurrent
from types import SimpleNamespace
from functools import partial
from typing import *

def num_cpus():
    """
    Get number of cpus
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def ifnone(a, b):
    """
    Return if None
    """
    return b if a is None else a
def parallel(func, arr: Collection, max_workers: int = 12, **kwargs):

    """
    Call `func` on every element of `arr` in parallel using `max_workers`.
    """
    _default_cpus = min(max_workers, num_cpus())
    defaults = SimpleNamespace(
        cpus=_default_cpus, cmap="viridis", return_fig=False, silent=False
    )

    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers < 2:
        results = [func(o) for i, o in tqdm(enumerate(arr), total=len(arr))]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func, o) for i, o in enumerate(arr)]
            results = []
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(arr)):
                results.append(f.result())
    if any([o is not None for o in results]):
        return results


MAX_THREAD = max(multiprocessing.cpu_count(),10) - 1
# MAX_THREAD = 12
print(MAX_THREAD)

parser = argparse.ArgumentParser()
parser.add_argument('--modelnet10', help="Specify root directory to the ModelNet10 dataset.", required=True)
parser.add_argument('--out', help='Specify folder to save output', default='.')
parser.add_argument('--n_voxels', help='Number of voxels per dimension.', default=50, type=int)
args = parser.parse_args()

VOXEL_SIZE = float(1 / args.n_voxels)
BASE_DIR = sys.path[0]
DATA_PATH = os.path.join(BASE_DIR, args.modelnet10)
VOX_DIR = os.path.join(BASE_DIR, ".voxel_data")

labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)
labels.sort()

# labels = labels[6::]

if os.path.exists(VOX_DIR):
    import shutil
    shutil.rmtree(VOX_DIR)
os.makedirs(VOX_DIR)
for lab in labels:
    os.makedirs(os.path.join(VOX_DIR, lab, 'train'))
    os.makedirs(os.path.join(VOX_DIR, lab, 'test'))

def run_train(file, label):
    try:
        filename = os.path.join(DATA_PATH, label, "train", file)
        print(f"Elaborating file {filename}...")
        out_name = os.path.join(VOX_DIR, label, 'train', file.split(".")[0] + ".npy")
        mesh = o3d.io.read_triangle_mesh(filename)
        # print(mesh)
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                center=mesh.get_center())
        center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
        mesh = mesh.translate((-center[0], -center[1], -center[2]))

        # (1/voxel_size)^3 will be the size of the input of the network, 0.02 results in 50^3=125000
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                                    min_bound=np.array(
                                                                                        [-0.5, -0.5, -0.5]),
                                                                                    max_bound=np.array([0.5, 0.5, 0.5]),
                                                                                    )
        voxels = voxel_grid.get_voxels()
        # print("voxel")
        grid_size = args.n_voxels
        mask = np.zeros((grid_size, grid_size, grid_size))
        for vox in voxels:
            mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
        np.save(out_name, mask, allow_pickle=False, fix_imports=False)
    except Exception as e:
        print(e)


def run_test(file,label):
    try:
        filename = os.path.join(DATA_PATH, label, "test", file)
        # print(f"Elaborating file {filename}...")
        out_name = os.path.join(VOX_DIR, label, 'test', file.split(".")[0] + ".npy")
        mesh = o3d.io.read_triangle_mesh(filename)
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                center=mesh.get_center())
        center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
        mesh = mesh.translate((-center[0], -center[1], -center[2]))

        # (1/voxel_size)^3 will be the size of the input of the network, 0.02 results in 50^3=125000
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                                    min_bound=np.array(
                                                                                        [-0.5, -0.5, -0.5]),
                                                                                    max_bound=np.array([0.5, 0.5, 0.5]), )
        voxels = voxel_grid.get_voxels()
        grid_size = args.n_voxels
        mask = np.zeros((grid_size, grid_size, grid_size))
        for vox in voxels:
            mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
        np.save(out_name, mask, allow_pickle=False, fix_imports=False)
    except:
        pass


for label in tqdm(labels, total=len(labels)):
    files_train = os.listdir(os.path.join(DATA_PATH, label, "train"))
    files_test = os.listdir(os.path.join(DATA_PATH, label, "test"))
    files_train.sort()
    files_test.sort()
    
    for file in files_train:
        if not file.endswith('off'):
            files_train.remove(file)
    for file in files_test:
        if not file.endswith('off'):
            files_test.remove(file)

    print(len(files_train), len(files_test))
    files_train = files_train[:100]
    # results = Parallel(n_jobs=MAX_THREAD)(delayed(run_train)(file, label) for file in files_train)

    # results = Parallel(n_jobs=MAX_THREAD)(delayed(run_train)(file, label) for file in files_test)
    results = parallel(partial(run_train, label = label), files_train, num_cpus = MAX_THREAD)
