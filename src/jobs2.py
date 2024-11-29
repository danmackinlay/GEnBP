""" A do-over of jobs.py to be more resumable and robust
"""
import time
from collections import defaultdict
from pprint import pprint
import warnings

import torch
from torch.autograd import grad
import numpy as np

import submitit
import bz2
import cloudpickle


import numpy as np
import submitit
import bz2
import cloudpickle
import os
import matplotlib.pyplot as plt


def submit_jobs(executor, fn, base_kwargs, sweep_param, sweep_values, n_replicates, experiment_name, job_info=None, batch=False):
    if job_info is None:
        job_info = []
    seed = 0
    executor.update_parameters(name=experiment_name)
    # hacky way of selecting map_array mode or not:
    if batch:
        with executor.batch():
            for value in sweep_values:
                for replicate in range(n_replicates):
                    kwargs = base_kwargs.copy()
                    kwargs[sweep_param] = value
                    kwargs['seed'] = seed
                    seed += 1
                    print(f"experiment_name: {experiment_name} {sweep_param}={value} replicate={replicate}")
                    job = executor.submit(fn, **kwargs)
                    job_info.append({'job': job, 'params': kwargs})
    else:
        for value in sweep_values:
            for replicate in range(n_replicates):
                kwargs = base_kwargs.copy()
                kwargs[sweep_param] = value
                kwargs['seed'] = seed
                seed += 1
                print(f"experiment_name: {experiment_name} {sweep_param}={value} replicate={replicate}")
                job = executor.submit(fn, **kwargs)
                job_info.append({'job': job, 'params': kwargs})

    return job_info


def collate_job_results(job_info, sweep_param):
    results = {}
    for info in job_info:
        job = info['job']
        params = info['params']
        sweep_value = params[sweep_param]
        print(f"waiting for {info}")
        try:
            job_result = job.result()
            if sweep_value not in results:
                results[sweep_value] = []
            results[sweep_value].append(job_result)
        except Exception as e:
            warnings.warn(f"Job {job} failed with state {job.state}, {job.stderr()}")

    # Sort results by sweep_param values
    sorted_results = {k: results[k] for k in sorted(results)}

    # Reduce results: transform list of dicts to dict of lists
    reduced_results = {k: {rk: [d[rk] for d in v] for rk in v[0]} for k, v in sorted_results.items() if v}
    return reduced_results


def save_artefact(artefact, file_path):
    """
    Zipped pickler that saves an artefact to a given file path,
    ensuring the existence of parent directories.
    """
    # Ensure the existence of parent directories
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Save the artefact to the specified file path
    with bz2.open(file_path, "wb") as f:
        cloudpickle.dump(artefact, f)
    return file_path


def load_artefact(file_path):
    with bz2.open(file_path, "rb") as f:
        return cloudpickle.load(f)


def construct_intermediate_path(path_fragment):
    """
    Constructs a file path for an artefact within the "_logs" directory,
    appending ".pkl.bz2" to the given path fragment.
    """
    base_dir = "_logs"
    filename = f"{path_fragment}.pkl.bz2"
    return os.path.join(base_dir, filename)


def construct_output_path(path_fragment):
    """
    Constructs a file path for an artefact within the "outputs" directory,
    appending ".pkl.bz2" to the given path fragment.
    """
    base_dir = "outputs"
    filename = f"{path_fragment}.pkl.bz2"
    return os.path.join(base_dir, filename)


def construct_figure_path(path_fragment):
    """
    Constructs a file path for an artefact within the "outputs" directory,
    appending ".pkl.bz2" to the given path fragment.
    """
    base_dir = "outputs"
    filename = f"{path_fragment}.pkl.bz2"
    return os.path.join(base_dir, filename)



def example_calc(a=1, b=2, seed=3):
    """
    for demonstrating purposes an example job we might wish to run in parallel
    """
    np.random.seed(seed)

    # Start timing and memory measurement
    start_time = time.time()

    # The part of the code you want to measure
    # Example calculation (replace with your actual code)
    result = a + b

    # End timing and memory measurement
    end_time = time.time()

    # Calculate elapsed time and peak memory usage
    elapsed_time = end_time - start_time

    return {
        'result': result,
        'time': elapsed_time,
    }

def run_example_sweep():
    import submitit

    # Example usage
    experiment_name = "my_experiment"
    sweep_param = 'a'  # The parameter to sweep over
    sweep_values = range(10)  # The range of values to sweep over
    n_replicates = 5  # Number of replicates for each value

    # Set up the executor
    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(timeout_min=1, slurm_partition="dev", tasks_per_node=4, mem=8)

    # Submit jobs
    base_kwargs = {}  # Base arguments for expensive_calc, if any
    job_info, path = submit_jobs(executor, base_kwargs, sweep_param, sweep_values, n_replicates, experiment_name=experiment_name)

if __name__ == "__main__":
    run_example_experiment()
