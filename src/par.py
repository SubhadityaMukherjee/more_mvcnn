"""
Some utlity scripts for parallel processing wherever needed
"""
import concurrent
import multiprocessing
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import *

from tqdm import tqdm


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
