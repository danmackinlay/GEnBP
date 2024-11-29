# %%
"""
sanity check the transport problem works ok
"""
# %load_ext autoreload
# %autoreload 2
# %run mp_026_sysid_advection_wrapped.py
from mp_026_sysid_advection_wrapped import *

import matplotlib.pyplot as plt
from tueplots import bundles, figsizes
# plt.rcParams.update(bundles.iclr2024())
# plt.rcParams.update(figsizes.iclr2024())

method = "genbp"
# method = "gabp"
# method = "laplace"


#%% sanity check
base_kwargs = dict(
    d=64,
    circ_radius=0.125,
    conv_radius=0.125,
    downsample=2,
    obs_sigma2=0.01,  # obs noise
    tau2=0.1,  # process noise
    decay=0.7,
    shift=10,
    n_timesteps=10,
    ## inference params
    method=method,
    n_ens=64,
    # damping=0.25,   # damping
    damping=0.0,  # no damping
    # gamma2=0.1,
    hard_damping=True,
    # inf_tau2=0.1,   # assumed process noise (so it can be >0)
    inf_gamma2=0.065,  # bp noise
    q_gamma2=None,  # bp noise
    inf_sigma2=0.069,  # inference noise
    inf_eta2=0.067,  # conformation noise
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
    ## diagnostics
    DEBUG_MODE=True,
    DEBUG_PLOTS=True,
    FINAL_PLOTS=True,
    SAVE_FIGURES=True,
    # return_fg=True,
    forney_mode=(method=='gbp'),
    job_name=f"adv_{method}_sanity",
    q_ylim=(-2,8),
    lw=0.2,
    alpha_scale=2.0,
)
#%%
# executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
# executor.update_parameters(
#     timeout_min=59,
#     # gpus_per_node=1,
#     slurm_account=os.getenv('SLURM_ACCOUNT'),
#     slurm_mem=8*1024,
# )
# j = executor.submit(run_run, **base_kwargs, seed=1)
# result = j.result()

# laplace_lik_result = run_run(**{**base_kwargs, **LAPLACE_BEST_LOGLIK, 'job_name': "adv_laplace_lik_sanity"}, seed=75)
# pprint(laplace_lik_result)

# gbp_lik_result = run_run(**{**base_kwargs, **GBP_BEST_LOGLIK, 'job_name': "adv_gbp_lik_sanity"}, seed=75)
# pprint(gbp_lik_result)
# genbp_lik_result = run_run(**{**base_kwargs, **GENBP_BEST_LOGLIK, 'job_name': "adv_genbp_lik_sanity"}, seed=75)
# pprint(genbp_lik_result)
# gbp_mse_result = run_run(**{**base_kwargs, **GBP_BEST_MSE, 'job_name': "adv_gbp_mse_sanity"}, seed=75)
# pprint(gbp_mse_result)
# genbp_mse_result = run_run(**{**base_kwargs, **GENBP_BEST_MSE, 'job_name': "adv_genbp_mse_sanity"}, seed=75)
# pprint(genbp_mse_result)

# %%
