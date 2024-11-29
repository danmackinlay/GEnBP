# %%
"""

"""
# %load_ext autoreload
# %autoreload 2
import bz2
import cloudpickle
import warnings
from pprint import pprint, pformat

# %run mp_026_sysid_advection_wrapped.py
from mp_026_sysid_advection_wrapped import *


#%% sweep some params

base_kwargs = dict(
    d=256,
    circ_radius=0.125,
    conv_radius=0.125,
    downsample=3,
    obs_sigma2=0.01,  # obs noise
    tau2=0.1,  # process noise
    decay=0.7,
    shift=6,
    n_timesteps=10,
    ## inference params
    # method='gbp',
    n_ens=100,
    # damping=0.25,   # damping
    damping=0.0,  # no damping
    # gamma2=0.1,
    hard_damping=True,
    # inf_tau2=0.1,   # assumed process noise (so it can be >0)
    inf_gamma2=0.5,  # bp noise
    q_gamma2=None,  # bp noise
    inf_sigma2=0.5,  # inference noise
    inf_eta2=0.1,  # conformation noise    max_rank=None,
    max_steps=150,
    rtol=1e-6,
    atol=1e-8,
    empty_inboxes=True,
    min_mp_steps=10,
    belief_retain_all=False,
    conform_retain_all=True,
    conform_r_eigen_floor=1e-4,
    conform_randomize=True,
    ## diagnostics
    DEBUG_MODE=False,
    DEBUG_PLOTS=False,
    FINAL_PLOTS=False,
    SAVE_FIGURES=False,
    return_fg=False,
    # job_name="adv_genbp_dev",
    q_ylim=(-3,6),
)


def submit_random_search_jobs(executor,fn, job_count, base_kwargs, job_name, param_distributions):
    jobs = []
    param_list = []  # List to store parameters for each job

    for i in range(job_count):
        kwargs = base_kwargs.copy()

        # Randomly sample from each distribution
        for param, dist in param_distributions.items():
            kwargs[param] = dist.sample().item()

        job = executor.submit(fn, **kwargs, seed=i)
        jobs.append(job)
        param_list.append(kwargs)

    # Save jobs and param_list to disk
    with bz2.open(job_name + ".job.pkl.bz2", "wb") as f:
        cloudpickle.dump((jobs, param_list), f)

    return jobs, param_list


def winnow_and_gather_job_results(jobs, param_list):
    successful_params = []
    successful_results = []
    failed_params = []
    total_jobs = len(jobs)
    failed_jobs = 0

    for job, params in zip(jobs, param_list):
        try:
            result = job.result()
            successful_params.append(params)
            successful_results.append(result)
        except Exception as e:
            failed_jobs += 1
            failed_params.append(params)
            warnings.warn(f"Job not completed successfully with parameters: {params} \n{e}")
            # print(job.stdout())
            print(job.stderr())

    job_statistics = {
        "total_jobs": total_jobs,
        "completed_jobs": total_jobs - failed_jobs,
        "failed_jobs": failed_jobs
    }

    return successful_params, successful_results, job_statistics, failed_params


def find_parameters_with_highest_lowest_metric(successful_params, successful_results, metric, highest=True):
    if not successful_results:
        raise ValueError("No completed jobs to analyze")

    # Combine parameters and results into a single list of tuples
    combined = list(zip(successful_params, successful_results))

    # Sort the combined list by the specified metric
    # highest=True for the highest value, highest=False for the lowest value
    sorted_combined = sorted(combined, key=lambda x: x[1][metric], reverse=highest)

    # Return the parameters and result with the highest or lowest metric value
    optimal_params, optimal_result = sorted_combined[0]
    return optimal_params, optimal_result

#%%
executor = submitit.AutoExecutor(folder=LOG_DIR)

# executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=59,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_mem=8*1024,
)

param_distributions = {
    'inf_sigma2': torch.distributions.Exponential(20),
    'inf_eta2': torch.distributions.Exponential(20),
    'inf_gamma2': torch.distributions.Exponential(20),
}
gbp_experiment_name = "advect_gbp_hyp"
genbp_experiment_name = "advect_genbp_hyp"

#%%

executor.update_parameters(name=gbp_experiment_name)

gbp_jobs, gbp_param_list = submit_random_search_jobs(executor, run_run, 200, {**base_kwargs, 'method': 'gbp'}, gbp_experiment_name, param_distributions)

#%%

executor.update_parameters(name=genbp_experiment_name)

genbp_jobs, genbp_param_list = submit_random_search_jobs(executor, run_run, 200, {**base_kwargs, 'method': 'genbp'}, genbp_experiment_name, param_distributions)
#%%
genbp_successful_params, genbp_successful_results, genbp_job_statistics, genbp_failed_params = winnow_and_gather_job_results(genbp_jobs, genbp_param_list)
with bz2.open("outputs/sysid_advection_genbp_hyp.pkl.bz2", "wb") as f:
    cloudpickle.dump((genbp_successful_params, genbp_successful_results, genbp_job_statistics, genbp_failed_params), f)
#%%
gbp_successful_params, gbp_successful_results, gbp_job_statistics, gbp_failed_params = winnow_and_gather_job_results(gbp_jobs, gbp_param_list)
with bz2.open("outputs/sysid_advection_gbp_hyp.pkl.bz2", "wb") as f:
    cloudpickle.dump((gbp_successful_params, gbp_successful_results, gbp_job_statistics, gbp_failed_params), f)
#%%
with bz2.open("outputs/sysid_advection_genbp_hyp.pkl.bz2", "rb") as f:
    genbp_successful_params, genbp_successful_results, genbp_job_statistics, genbp_failed_params = cloudpickle.load(f)
with bz2.open("outputs/sysid_advection_gbp_hyp.pkl.bz2", "rb") as f:
    gbp_successful_params, gbp_successful_results, gbp_job_statistics, gbp_failed_params = cloudpickle.load(f)
#%%
genbp_best_mse_param, genbp_best_mse = find_parameters_with_highest_lowest_metric(genbp_successful_params, genbp_successful_results, 'q_mse', highest=False)
print(f"GENBP_BEST_MSE = {pformat(genbp_best_mse)}\nat\n{pformat(genbp_best_mse_param)}",)
genbp_best_loglik_param, genbp_best_loglik = find_parameters_with_highest_lowest_metric(genbp_successful_params, genbp_successful_results, 'q_loglik', highest=True)
print(f"GENBP_BEST_LOGLIK = {pformat(genbp_best_loglik)}\nat\n{pformat(genbp_best_loglik_param)}",)
gbp_best_mse_param, gbp_best_mse = find_parameters_with_highest_lowest_metric(gbp_successful_params, gbp_successful_results, 'q_mse', highest=False)
print(f"GBP_BEST_MSE = {pformat(gbp_best_mse)}\n{pformat(gbp_best_mse_param)}",)
gbp_best_loglik_param, gbp_best_loglik = find_parameters_with_highest_lowest_metric(gbp_successful_params, gbp_successful_results, 'q_loglik', highest=True)
print(f"GBP_BEST_LOGLIK = {pformat(gbp_best_loglik)}\nat\n{pformat(gbp_best_loglik_param)}",)

#%%
#scatter-plot the loglik versus mse for the genbp case

import matplotlib.pyplot as plt
from tueplots import bundles, figsizes
plt.rcParams.update(bundles.iclr2024())
plt.rcParams.update(figsizes.iclr2024())
mses = np.array([r['q_mse'] for r in genbp_successful_results])
logliks =  np.array([r['q_loglik'] for r in genbp_successful_results])
# filter out loglik outliers
#first find percentiles
loglik_percentiles = np.nanpercentile(logliks, [0, 5, 10, 25, 50, 75, 90, 95, 100])
print(loglik_percentiles)
# The bottom 25% are on a different scale


fig, axs = plt.subplots()
plt.scatter(
    mses[inliers],
    logliks[inliers], s=0.5)
# label axes
plt.xlabel('Mean-Squared Error')
plt.ylabel('Log Likelihood')
# lims chosen for this data set
plt.xlim([0.0, 0.12])
plt.ylim([-100, 220])
# save figure
plt.savefig('fig/genbp_hyp_mse_vs_loglik.pdf')
# %%
