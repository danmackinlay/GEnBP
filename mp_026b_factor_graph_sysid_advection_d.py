# %%
"""
Sweep dimension to ensure good behavious
"""
# %load_ext autoreload
# %autoreload 2
# %run mp_026_factor_graph_sysid_advection_wrapped.py
from mp_026_factor_graph_sysid_advection_wrapped import *


# genbp_experiment_name = "advect_genbp_d_sweep"
# gbp_experiment_name = "advect_gbp_d_sweep"
# laplace_experiment_name = "advect_laplace_d_sweep"
genbp_experiment_name = "adv_genbp_d_iclr"
gbp_experiment_name = "adv_gbp_d_iclr"
laplace_experiment_name = "adv_laplace_d_iclr"
sweep_param = 'd'


#%% sweep some params
base_kwargs = dict(
    d=512,
    circ_radius=0.125,
    conv_radius=0.125,
    downsample=2,
    obs_sigma2=0.01,  # obs noise
    tau2=0.1,  # process noise
    decay=0.7,
    shift=10,
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
    SAVE_FIGURES=True,
    SHOW_FIGURES=False,
    return_fg=False,
    # job_name="adv_genbp_dev",
    # q_ylim=(-3,6),
)

executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=59,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_array_parallelism=50,
    slurm_mem=8*1024,
)
x = np.arange(16, 512, 32)
# sweep_values = np.geomspace(0.1, 100, num=10)  # Exponential sweep for 'a'
# sweep_values = np.arange(16, 512, 32)
sweep_values = x.astype(int)

n_replicates = 40

gbp_experiment = sweep_params(
    run_run,
    {**base_kwargs,
    'method':'gbp', 'job_name': gbp_experiment_name},
    sweep_param, sweep_values,
    n_replicates=n_replicates,
    executor=executor, experiment_name=gbp_experiment_name,
    log_dir=LOG_DIR, batch=True
)

genbp_experiment = sweep_params(
    run_run,
    {**base_kwargs, 'method':'genbp', 'job_name': genbp_experiment_name},
    sweep_param, sweep_values,
    n_replicates=n_replicates, executor=executor, experiment_name=genbp_experiment_name,
    log_dir=LOG_DIR, batch=True
)
laplace_experiment = sweep_params(
    run_run,
    {**base_kwargs, 'method':'laplace', 'job_name': laplace_experiment_name},
    sweep_param, sweep_values,
    n_replicates=n_replicates, executor=executor, experiment_name=laplace_experiment_name,
    log_dir=LOG_DIR, batch=True
)
#%% resume experiment
gbp_experiment = load_experiment(gbp_experiment_name, LOG_DIR)
genbp_experiment = load_experiment(genbp_experiment_name, LOG_DIR)
laplace_experiment = load_experiment(laplace_experiment_name, LOG_DIR)
#%%
gbp_experiment_results = reduce_experiment(
    gbp_experiment,
    lambda a: compute_percentiles(a, percentiles=[0.1, 0.5, 0.9]))
pprint(gbp_experiment_results)

print("saving to", save_artefact(gbp_experiment_results, gbp_experiment_name, OUTPUT_DIR))

# %%
genbp_experiment_results = reduce_experiment(
    genbp_experiment,
    lambda a: compute_percentiles(a, percentiles=[0.1, 0.5, 0.9]))
pprint(genbp_experiment_results)

print("saving to", save_artefact(genbp_experiment_results, genbp_experiment_name, OUTPUT_DIR))
# %%
laplace_experiment_results = reduce_experiment(
    laplace_experiment,
    lambda a: compute_percentiles(a, percentiles=[0.1, 0.5, 0.9]))
pprint(laplace_experiment_results)

print("saving to", save_artefact(laplace_experiment_results, laplace_experiment_name, OUTPUT_DIR))
# %%
gbp_experiment_results = load_artefact(gbp_experiment_name, OUTPUT_DIR)
genbp_experiment_results = load_artefact(genbp_experiment_name, OUTPUT_DIR)
laplace_experiment_results = load_artefact(laplace_experiment_name, OUTPUT_DIR)
y_keys = [
    'time',
    # 'memory',
    'q_mse',
    'q_loglik'
]
titles = [
    'Execution Time (s)',
    # 'Memory Usage',
    r'Mean-Squared Error ',
    r'Log Likelihood'
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
# We create several axes  to plot all experiments for time, memory, MSE and likelihood.
from tueplots import bundles, figsizes
import matplotlib as mpl

n_plots = len(y_keys)
mode="row"
plt.rcParams.update(bundles.iclr2024())
plt.rcParams['text.latex.preamble'] = plt.rcParams['text.latex.preamble'] + r'\usepackage{mathrsfs}'
if mode == "col":
    plt.rcParams.update(figsizes.iclr2024(nrows=n_plots, ncols=1))
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)
else:
    plt.rcParams.update(figsizes.iclr2024(ncols=n_plots, nrows=1, height_to_width_ratio=1.0))
    fig, axs = plt.subplots(ncols=n_plots, nrows=1)

offsets = np.array([-1.,0,1.]) * 2

for i, ax in enumerate(axs):
    y_key = y_keys[i]
    title = titles[i]
    title_position = title_positions[i]
    better_high = better_high_dict[y_key]

    plot_experiment_results(
        ax,
        sweep_param,
        y_key,
        genbp_experiment_results,
        color="green",
        label="GEnBP",
        x_offset=offsets[0]
    )
    plot_experiment_results(
        ax,
        sweep_param,
        y_key,
        gbp_experiment_results,
        color='blue',
        label='GaBP',
        x_offset=offsets[1]
    )
    plot_experiment_results(
        ax,
        sweep_param,
        y_key,
        laplace_experiment_results,
        color='brown',
        label='Laplace',
        x_offset=offsets[2]
    )

    if i == 0:
        ax.legend(loc='best', fontsize='small')

    if mode == "col":
        ax.set_ylabel(title)
        # Set arrow properties
        # arrow_color = 'green' if better_high else 'red'
        arrow_color = 'black'
        arrow_direction = '<-' if better_high else '->'

        # Place the arrow outside the plot in the left margin
        # `transform=ax.transAxes` means position is relative to the axes boundary, not the data
        # Calculate starting and ending points for the arrow
        base_y = 0.05
        head_y = 0.15

        # Place the arrow outside the plot in the left margin
        # Align the low end of the arrow with the bottom axis of the plot
        ax.annotate(
            '',
            xy=(-0.15, base_y),
            xycoords='axes fraction',
            xytext=(-0.15, base_y + head_y),
            textcoords='axes fraction',
            arrowprops=dict(
                arrowstyle=f"{arrow_direction}, head_width=0.4, head_length=0.6",
                facecolor=arrow_color))
    else:
        arrow = r'$\uparrow$' if better_high else r'$\downarrow$'

        ax.set_title(title + " " + arrow, pad=15)


    if (i == len(axs) - 1) or mode == "row":
        ax.set_xlabel(r"$D_\mathscr{Q}$")
    if mode == "row":
        ax.text(-0.0, -0.2, f"{chr(97+i)})", transform=ax.transAxes, ha='left', va='top')

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
    os.path.join(FIG_DIR, f'adv_d_sweep.pdf'), bbox_inches='tight')
plt.show()

# %%
