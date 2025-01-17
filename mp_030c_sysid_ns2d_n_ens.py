# %%
"""Let's sweep ensemble size
"""
# %load_ext autoreload
# %autoreload 2
from src.jobs import *
# %run mp_030_sysid_ns2d_wrapped.py
from mp_030_sysid_ns2d_wrapped import *

genbp_experiment_name = "cfd_genbp_n_ens_iclr"
sweep_param = 'n_ens'

#%% sweep some params
base_kwargs = dict(
    d=32,
    visc = 0.01,
    delta_t = 0.2,
    interval = 5,
    v_noise_power = 0.0,
    downsample=5,
    obs_sigma2=0.01,  # obs noise
    n_timesteps=4,
    ## inference params
    method='genbp',
    cvg_tol=-1.0,
    n_ens=64,
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
    DEBUG_MODE=False,
    DEBUG_PLOTS=False,
    FINAL_PLOTS=False,
    SAVE_FIGURES=False,
    return_fg=False,
    # job_name="cfd_genbp_dev",
    q_ylim=(-1,1),
)

executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=59,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_array_parallelism=50,
    slurm_mem=16*1024,
)
executor.update_parameters(name=genbp_experiment_name)

x = np.linspace(16, 256, 16)
# sweep_values = np.geomspace(0.1, 100, num=10)  # Exponential sweep for 'a'
# sweep_values = np.arange(16, 256, 16)
sweep_values = x.astype(int)

n_replicates = 40

genbp_experiment = sweep_params(
    run_run,
    {**base_kwargs, 'method':'genbp', 'job_name': genbp_experiment_name},
    sweep_param, sweep_values,
    n_replicates=n_replicates, executor=executor, experiment_name=genbp_experiment_name,
    log_dir=LOG_DIR, batch=True
)
#%% resume experiment
genbp_experiment = load_experiment(genbp_experiment_name, LOG_DIR)

# %%
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
    'Mean-Squared Error',
    'Log Likelihood'
]
title_positions = [
    'bottom',
    # 'bottom',
    'top',
    'bottom'
]
better_high_dict = {
    'time': False,
    'q_mse': False,
    'q_loglik': True,
}
# Now we create a plot of the performance of each method.
# We create several axes  to reuse to plot both experiments for time, memory, MSE and likelihood.
from tueplots import bundles, figsizes
import matplotlib as mpl
n_plots = len(y_keys)
mode = "row"
plt.rcParams.update(bundles.iclr2024())
plt.rcParams['text.latex.preamble'] = plt.rcParams['text.latex.preamble'] + r'\usepackage{mathrsfs}'
if mode == "col":
    plt.rcParams.update(figsizes.iclr2024(nrows=n_plots, ncols=1))
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)
else:
    plt.rcParams.update(figsizes.iclr2024(ncols=n_plots, nrows=1, height_to_width_ratio=1.0))
    fig, axs = plt.subplots(ncols=n_plots, nrows=1)

for i, ax in enumerate(axs):
    y_key = y_keys[i]
    title = titles[i]
    title_position = title_positions[i]

    plot_experiment_results(
        ax, sweep_param, y_key, genbp_experiment_results,
        color='green', label='GEnBP',
        # show_legend=(i == 0),
        # show_labels=(i == len(axs) - 1)
        # show_labels=False
    )

    # Add title inside the plot area
    if mode=="col":
        vert_align = 'top' if title_position == 'top' else 'bottom'
        horiz_align = 'right'
        x_pos = 0.95
        y_pos = 0.95 if title_position == 'top' else 0.05
        ax.text(
            x_pos, y_pos, title,
            transform=ax.transAxes, fontsize=10, va=vert_align, ha=horiz_align,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
    else:
        ax.set_title(title)

    if y_key in (
            "time",
            # "q_mse"
            ):
        ax.set_yscale('log')
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0e}'.format(y)))
        ax.yaxis.set_major_formatter(
            mpl.ticker.LogFormatterSciNotation(labelOnlyBase=True))
        # ax.ticklabel_format(
        #     style='scientific',
        #     axis='y',
        #     scilimits=(0,0)
        # )
    else:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1,1))

# plt.tight_layout()
# save figure as PDF in FIG_DIR
plt.savefig(
    os.path.join(FIG_DIR, f'cfd_{sweep_param}_sweep.pdf'), bbox_inches='tight')
plt.show()

# %%
