# %%
"""Baseline model inference functionality
"""
%load_ext autoreload
%autoreload 2
# %run mp_030_factor_graph_sysid_ns2d_wrapped.py
from mp_030_factor_graph_sysid_ns2d_wrapped import *

# Choose the inference method
# method = 'genbp'
# method = 'gbp'
# method = 'laplace'
method = 'langevin'  # Set method to 'langevin' to use the Langevin sampler

# %% Sanity check
base_kwargs = dict(
    d=32,
    visc=0.01,
    delta_t=0.2,
    interval=5,
    v_noise_power=0.0,
    downsample=5,
    obs_sigma2=0.01,  # Observation noise variance
    n_timesteps=8,
    ## Inference parameters
    method=method,
    n_ens=200,
    cvg_tol=-1.0,
    damping=0.0,  # No damping
    hard_damping=True,
    inf_gamma2=0.01,  # BP noise
    q_gamma2=0.01,    # BP noise
    inf_sigma2=0.001,  # Inference noise
    inf_eta2=0.1,     # Conformation noise
    max_rank=None,
    max_steps=150,
    rtol=1e-6,
    atol=1e-8,
    empty_inboxes=True,
    min_mp_steps=10,
    belief_retain_all=False,
    conform_retain_all=True,
    conform_r_eigen_floor=1e-4,
    conform_randomize=True,
    ## Diagnostics
    DEBUG_MODE=True,
    DEBUG_PLOTS=True,
    FINAL_PLOTS=True,
    SAVE_FIGURES=False,
    return_fg=True,
    forney_mode=True,
    q_ylim=(-1, 1),
    lw=0.2,
    # alpha_scale=2.0,
    # langevin parameters
    langevin_step_size= 0.001,
    langevin_num_samples= 5000,
    langevin_burn_in= 1000,
    langevin_thinning= 10,
)

# You can use submitit for job submission if desired
# executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
# executor.update_parameters(
#     timeout_min=59,
#     # gpus_per_node=1,
#     slurm_account=os.getenv('SLURM_ACCOUNT'),
#     slurm_mem=8*1024,
# )
# j = executor.submit(run_run, **base_kwargs, job_name=f"fg_cfd_{method}_forney_sanity", seed=1)
# result = j.result()

# Direct invocation of the run_run function
result = run_run(
    **base_kwargs,
    job_name=f"fg_cfd_{method}_forney_sanity",
    seed=74
)

# Print the results
from pprint import pprint
pprint(result)

# %%
