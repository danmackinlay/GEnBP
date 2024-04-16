# %%
"""Inference in a model like this

       X0
       |
       v
Q----->X1---->X1
|      |      |
|      v      v
\----->X2---->Y2
|      |      |
|      v      v
\----->X3---->Y3
...

where the factors are high dimensional.
The target is Q, X0 is known.
"""
# %load_ext autoreload
# %autoreload 2
%run mp_026_factor_graph_sysid_advection_wrapped.py
genbp_experiment_name = "fg_advect_genbp_n_ens_sweep"
sweep_param = 'n_ens'


#%% sweep some params
base_kwargs = dict(
    d=512,
    circ_radius=0.125,
    conv_radius=0.125,
    downsample=2,
    obs_sigma2=0.01,  # obs noise
    tau2=0.1,  # process noise
    decay=0.7,
    shift=16,
    n_timesteps=6,
    ## inference params
    # method='gbp',
    n_ens=64,
    # damping=0.25,   # damping
    damping=0.0,  # no damping
    # gamma2=0.1,
    hard_damping=True,
    # inf_tau2=0.1,   # assumed process noise (so it can be >0)
    inf_gamma2=0.01,  # bp noise
    inf_sigma2=0.01,  # inference noise
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
    DEBUG_MODE=False,
    DEBUG_PLOTS=False,
    FINAL_PLOTS=False,
    SAVE_FIGURES=False,
    return_fg=False,
    # job_name="fg_advect_genbp_dev",
    q_ylim=(-3,6),
)

executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=59,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_array_parallelism=20,
    slurm_mem=8*1024,
)
executor.update_parameters(name=genbp_experiment_name)

x = np.linspace(16, 512, 16)
# sweep_values = np.geomspace(0.1, 100, num=10)  # Exponential sweep for 'a'
sweep_values = np.arange(16, 256, 16)
sweep_values = x.astype(int)

n_replicates = 80


genbp_experiment = sweep_params(
    run_run,
    {**base_kwargs, 'method':'genbp', 'job_name': genbp_experiment_name},
    sweep_param, sweep_values,
    n_replicates=n_replicates, executor=executor, experiment_name=genbp_experiment_name,
    log_dir=LOG_DIR, batch=True
)
#%% resume experiment
genbp_experiment = load_experiment(genbp_experiment_name, LOG_DIR)

genbp_experiment_results = reduce_experiment(
    genbp_experiment,
    lambda a: compute_percentiles(a, percentiles=[0.1, 0.5, 0.9]))
pprint(genbp_experiment_results)

save_artefact(genbp_experiment_results, genbp_experiment_name, OUTPUT_DIR)
# %%
genbp_experiment_results = load_artefact(genbp_experiment_name, OUTPUT_DIR)

y_keys = [
    'time',
    # 'memory',
    'q_mse',
    'q_loglik'
]
titles = [
    'Execution Time',
    # 'Memory Usage',
    'MSE',
    'Log Likelihood'
]
title_positions = [
    'bottom',
    # 'bottom',
    'top',
    'bottom'
]

# Now we create a plot of the performance of each method.
# We create several axes  to reuse to plot both experiments for time, memory, MSE and likelihood.
from tueplots import bundles, figsizes
n_plots = len(y_keys)
plt.rcParams.update(bundles.icml2024())
plt.rcParams.update(figsizes.icml2024_half(nrows=n_plots, ncols=1))

fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)

for i, ax in enumerate(axs):
    y_key = y_keys[i]
    title = titles[i]
    title_position = title_positions[i]

    plot_experiment_results(
        ax, sweep_param, y_key, genbp_experiment_results, 'GEnBP', color='green',
        # show_legend=(i == 0),
        # show_labels=(i == len(axs) - 1)
        # show_labels=False
    )

    # Add title inside the plot area
    if title:
        vert_align = 'top' if title_position == 'top' else 'bottom'
        horiz_align = 'right'
        x_pos = 0.95
        y_pos = 0.95 if title_position == 'top' else 0.05
        ax.text(
            x_pos, y_pos, title,
            transform=ax.transAxes, fontsize=10, va=vert_align, ha=horiz_align,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

# plt.tight_layout()
# save figure as PDF in FIG_DIR
plt.savefig(
    os.path.join(FIG_DIR, f'fg_advect_n_ens_sweep.pdf'), bbox_inches='tight')
plt.show()

# %%
