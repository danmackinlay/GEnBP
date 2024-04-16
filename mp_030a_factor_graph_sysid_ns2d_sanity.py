# %%
"""Baseline model inference functionality
"""
%load_ext autoreload
%autoreload 2
%run mp_030_factor_graph_sysid_ns2d_wrapped.py

# method = 'genbp'
method = 'gbp'
#%% sanity check
base_kwargs = dict(
    d=32,
    visc = 0.01,
    delta_t = 0.2,
    interval = 5,
    v_noise_power = 0.0,
    downsample=5,
    obs_sigma2=0.01,  # obs noise
    n_timesteps=8,
    ## inference params
    method=method,
    n_ens=200,
    cvg_tol=-1.0,
    # damping=0.25,   # damping
    damping=0.0,  # no damping
    # gamma2=0.1,
    hard_damping=True,
    # inf_tau2=0.1,   # assumed process noise (so it can be >0)
    inf_gamma2=0.01,  # bp noise
    q_gamma2=0.01,  # bp noise
    inf_sigma2=0.001,  # inference noise
    inf_eta2=0.1,  # conformation noise
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
    SAVE_FIGURES=False,
    return_fg=True,
    forney_mode=True,
    q_ylim=(-1,1),
    lw=0.2,
    # alpha_scale=2.0,
)
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
result = run_run(
    **base_kwargs,
    job_name=f"fg_cfd_{method}_forney_sanity",
    seed=74)
# fg = result['fg']
pprint(result)

# %%
