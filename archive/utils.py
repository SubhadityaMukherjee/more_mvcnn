import numpy as np
from skimage.feature import peak_local_max
import os
import shutil
import datetime
import open3d as o3d
import copy
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
import concurrent
from types import SimpleNamespace
from functools import partial
from typing import *
import os
from tqdm import tqdm


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return o3d.utility.Vector3dVector(np_normalized)

def apply_noise(pcd, mu = 0.0, sigma= 1):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def int_to_1hot(n, dim):
    vec = np.zeros(dim)
    vec[n] = 1
    return vec


def view_vector(data, dim):
    res = np.zeros(dim)
    for n in data:
        res[n] = 1
    return res


def make_dir(path, delete=False):
    if os.path.exists(path) and delete is True:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


def extract_labels(data):
    mat = np.zeros((60, 1))
    for i in range(0, 60):
        entropy = float(data[data['view_code'] == i].entropy)
        mat[i] = entropy
    mat.resize((5, 12))
    coords = peak_local_max(mat, min_distance=1, exclude_border=False)
    labels = []
    for (y, x) in coords:
        labels.append((y * 12) + x)
    # fig, ax = plt.subplots(1)
    # ax.imshow(mat, cmap='rainbow')
    # for i in range(len(coords)):
    #     circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
    #     ax.add_patch(circle)
    #
    # plt.xticks([i for i in range(12)], [i*30 for i in range(12)])
    # plt.yticks([i for i in range(5)], [(i+1) * 30 for i in range(5)])
    # plt.show()

    return labels


def get_labels_from_object_views(data):
    subset_labels = extract_labels(data)
    # subset_idx = []
    # for lab in subset_labels:
    #     subset_idx.append(label2idx[lab])
    subset_labels.sort()
    return subset_labels


def get_label_dict(CLASSES, inverse=False):
    CLASSES.sort()
    # label2int = {'bathtub': 0,
    #              'bed': 1,
    #              'chair': 2,
    #              'desk': 3,
    #              'dresser': 4,
    #              'monitor': 5,
    #              'night_stand': 6,
    #              'sofa': 7,
    #              'table': 8,
    #              'toilet': 9}
    label2int = {x:id for id,x in enumerate(CLASSES)}
    int2label = {id:x for id,x in enumerate(CLASSES)}
    # print(label2int, int2label)

    # int2label = {0: 'bathtub',
    #              1: 'bed',
    #              2: 'chair',
    #              3: 'desk',
    #              4: 'dresser',
    #              5: 'monitor',
    #              6: 'night_stand',
    #              7: 'sofa',
    #              8: 'table',
    #              9: 'toilet'}
    if inverse:
        return int2label
    else:
        return label2int

# get_label_dict()

def get_datastamp():
    time = datetime.datetime.now()
    return time.strftime("%d-%b-%H%M%S")


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

