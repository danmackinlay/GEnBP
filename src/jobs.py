
import time
from memory_profiler import memory_usage
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


# Function to compute percentiles of return dict floats
def compute_percentiles(results, percentiles=[0.025, 0.5, 0.975]):
    percentile_results = {}
    keys = results[0].keys()
    for key in keys:
        try:
            values = [result[key] for result in results]
            percentile_results[key] = np.nanpercentile(values, [p * 100 for p in percentiles])
        except Exception as e:
            # re-raise but tell us which key failed.
            raise ValueError(f"Failed to compute percentiles for key {key}") from e
    return percentile_results

# Submit multiple runs of a function with the same parameters but varied seed
def run_trial(fn, trial_params, n_replicates, executor, batch=False):
    jobs = []
    if batch:
        with executor.batch():
            for run in range(n_replicates):
                seed = run
                job = executor.submit(fn, **trial_params, seed=seed)
                jobs.append(job)
    else:
        for run in range(n_replicates):
            seed = run  # Unique seed for each run
            job = executor.submit(fn, **trial_params, seed=seed)
            jobs.append(job)

    return jobs


def reduce_trial_jobs(jobs, reducer):
    results = []
    for job in jobs:
        job.wait()
        if job.state in ('DONE', 'COMPLETED'):
            results.append(job.result())
        else:
            warnings.warn("Job not completed")
            print(job.stdout())
            print(job.stderr())

    if len(results) == 0:
        raise ValueError("Empty trial")

    return reducer(results)


def submit_random_search_jobs(
        executor, job_count, base_kwargs, job_name, param_distributions):
    """
    # Example usage
    executor = ...  # Your executor setup here
    base_kwargs = {'a': 1, 'b': 2}  # Base kwargs for expensive_calc
    param_distributions = {
        'x': torch.distributions.Normal(0, 1),  # Normal distribution for 'x'
        'y': torch.distributions.Uniform(0, 5),  # Uniform distribution for 'y'
    }

    jobs, param_list = submit_random_search_jobs(executor, 100, base_kwargs, "my_random_search_job", param_distributions)
    """
    jobs = []
    param_list = []  # List to store parameters for each job

    for i in range(job_count):
        kwargs = base_kwargs.copy()

        # Randomly sample from each distribution
        for param, dist in param_distributions.items():
            kwargs[param] = dist.sample().item()

        job = executor.submit(expensive_calc, **kwargs, seed=i)
        jobs.append(job)
        param_list.append(kwargs)

    # Save jobs and param_list to disk
    with bz2.open(job_name + ".job.pkl.bz2", "wb") as f:
        cloudpickle.dump((jobs, param_list), f)

    return jobs, param_list


def gather_job_results(
        jobs, param_list):
    """
    # Example usage
    # Assuming you have a list of jobs and param_list from submit_random_search_jobs
    successful_params, successful_results, job_stats, failed_params = gather_job_results(jobs, param_list)
    print("Job statistics:", job_stats)
    """
    successful_params = []
    successful_results = []
    failed_params = []
    total_jobs = len(jobs)
    failed_jobs = 0

    for job, params in zip(jobs, param_list):
        job.wait()
        if job.state in ('DONE', 'COMPLETED'):
            successful_params.append(params)
            successful_results.append(job.result())
        else:
            failed_jobs += 1
            failed_params.append(params)
            warnings.warn("Job not completed")
            print(job.stdout())
            print(job.stderr())

    job_statistics = {
        "total_jobs": total_jobs,
        "completed_jobs": total_jobs - failed_jobs,
        "failed_jobs": failed_jobs
    }

    return successful_params, successful_results, job_statistics, failed_params


def find_parameters_with_highest_lowest_metric(successful_params, successful_results, metric, highest=True):
    """
    # Example usage
    params_with_highest_metric, highest_metric_result = find_parameters_with_highest_lowest_metric(successful_params, successful_results, 'mse', highest=True)
    print("Parameters with highest metric:", params_with_highest_metric)
    print("Result with highest metric:", highest_metric_result)
    """
    # Combine parameters and results into a single list of tuples
    combined = list(zip(successful_params, successful_results))

    # Sort the combined list by the specified metric
    # highest=True for the highest value, highest=False for the lowest value
    sorted_combined = sorted(combined, key=lambda x: x[1][metric], reverse=highest)

    # Return the parameters and result with the highest or lowest metric value
    optimal_params, optimal_result = sorted_combined[0]
    return optimal_params, optimal_result


def sweep_params(fn, base_kwargs, sweep_param, sweep_values, n_replicates, executor, experiment_name, log_dir, batch=False):
    executor.update_parameters(name=experiment_name)
    trials = []
    for value in sweep_values:
        trial_params = base_kwargs.copy()
        trial_params[sweep_param] = value
        jobs = run_trial(fn, trial_params, n_replicates, executor, batch=batch)
        trials.append({'params': trial_params, 'jobs': jobs})

    save_experiment(trials, experiment_name, log_dir=log_dir)
    return trials


def reduce_experiment(trials, reducer):
    experiment_results = []
    for trial in trials:
        try:
            trial_stats = reduce_trial_jobs(trial['jobs'], reducer)
            experiment_results.append({
                'params': trial['params'],
                'results': trial_stats
            })
        except ValueError:
            warnings.warn("Null trial")
            pprint(trial)
    return experiment_results


def save_experiment(experiment, experiment_name, log_dir):
    "zipped pickler, defaulting to a log directory"
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, experiment_name + ".experiment.pkl.bz2")
    with bz2.open(file_path, "wb") as f:
        return cloudpickle.dump(experiment, f)


def load_experiment(experiment_name, log_dir):
    "zipped unpickler, defaulting to a log directory"
    file_path = os.path.join(log_dir, experiment_name + ".experiment.pkl.bz2")
    with bz2.open(file_path, "rb") as f:
        return cloudpickle.load(f)


def save_artefact(artefact, artefact_name, output_dir):
    "zipped pickler, defaulting to an output directory"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, artefact_name + ".pkl.bz2")
    with bz2.open(file_path, "wb") as f:
        return cloudpickle.dump(artefact, f)


def load_artefact(artefact_name, output_dir):
    "zipped unpickler, defaulting to a log directory"
    file_path = os.path.join(output_dir, artefact_name + ".pkl.bz2")
    with bz2.open(file_path, "rb") as f:
        return cloudpickle.load(f)


# Process trial
def reduce_trial_jobs(jobs, fn):
    for job in jobs:
        job.wait()

    # Collect results
    results = []
    for job in jobs:
        if job.state in ('DONE', 'COMPLETED'):
            # pprint(job.result())
            results.append(job.result())
        else:
            warnings.warn("job not completed")
            print(job.stdout())
            print(job.stderr())

    if len(results)==0:
        raise ValueError("empty trial")

    # process each sweep value
    return fn(results)


def prepare_data_for_plotting(experiment_results, sweep_param):
    # Initialize a dictionary to hold arrays for each key in the results
    plot_data = defaultdict(lambda: defaultdict(list))

    for trial in experiment_results:
        param_value = trial['params'][sweep_param]
        for key, quantiles in trial['results'].items():
            for quantile_index, quantile_value in enumerate(quantiles):
                plot_data[key][quantile_index].append((param_value, quantile_value))

    # Convert lists to numpy arrays for easier plotting
    for key in plot_data:
        for quantile_index in plot_data[key]:
            plot_data[key][quantile_index] = np.array(plot_data[key][quantile_index])

    return plot_data


def plot_experiment_results(
        ax, sweep_param, y_key, experiment_results, label,
        **kwargs):
    plot_data = prepare_data_for_plotting(experiment_results, sweep_param)

    if y_key not in plot_data:
        raise ValueError(f"y_key '{y_key}' not found in experiment results.")

    # Extract parameter values, medians, and lower/upper percentiles
    param_values, lower_percentiles = zip(*plot_data[y_key][0])  # 0.025 nanpercentile
    _, medians = zip(*plot_data[y_key][1])  # 0.5 nanpercentile (median)
    _, upper_percentiles = zip(*plot_data[y_key][2])  # 0.975 nanpercentile

    # Calculate error for error bars (distance from median)
    lower_errors = np.array(medians) - np.array(lower_percentiles)
    upper_errors = np.array(upper_percentiles) - np.array(medians)
    error = [lower_errors, upper_errors]

    # Plotting
    ax.errorbar(
        param_values, medians, yerr=error, fmt='_',
        # capsize=5,
        label=label, **kwargs)

    return ax


def example_calc(a=1, b=2, seed=3):
    """
    for demonstrating purposes an example job we might wish to run in parallel
    """
    np.random.seed(seed)

    # Start timing and memory measurement
    start_time = time.time()
    peak_memory_start = memory_usage(max_usage=True)

    # The part of the code you want to measure
    # Example calculation (replace with your actual code)
    result = a + b

    # End timing and memory measurement
    end_time = time.time()
    peak_memory_end = memory_usage(max_usage=True)

    # Calculate elapsed time and peak memory usage
    elapsed_time = end_time - start_time
    peak_memory_usage = peak_memory_end - peak_memory_start

    return {
        'result': result,
        'time': elapsed_time,
        'memory': peak_memory_usage
    }


def run_example_experiment():
    """
    Example usage of multirun_sweep
    """
    from dotenv import load_dotenv
    load_dotenv()

    log_dir = os.getenv("LOG_DIR", "_logs")
    # Setup for the executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        timeout_min=59,
        # gpus_per_node=1,
        slurm_account=os.getenv('SLURM_ACCOUNT')
    )
    experiment_name = "my_test_job"
    base_kwargs = {'b': 2}  # Base kwargs
    sweep_param = 'a'
    sweep_values = np.geomspace(0.1, 100, num=10)  # Exponential sweep for 'a'
    n_replicates = 5  # Number of trials for each sweep value
    experiment = sweep_params(
        example_calc, base_kwargs, sweep_param, sweep_values, n_replicates, executor, experiment_name, log_dir)

    # Resume experiment
    experiment = load_experiment(experiment_name, log_dir)

    # Process results
    experiment_results = reduce_experiment(
        experiment,
        lambda results: compute_percentiles(results, percentiles=[0.025, 0.5, 0.975]))
    pprint(experiment_results)

if __name__ == "__main__":
    run_example_experiment()
