# %%
"""Performance in high dimension
"""
# %load_ext autoreload
# %autoreload 2
%run mp_030_factor_graph_sysid_ns2d_wrapped.py

from src import jobs2
from src.jobs import *
exp_prefix = "fg_cfd_dev"
sweep_param = 'd'

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
    # damping=0.25,   # damping
    damping=0.0,  # no damping
    hard_damping=True,
    n_ens=64,
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
    forney_mode=True,
    ## diagnostics
    DEBUG_MODE=False,
    DEBUG_PLOTS=False,
    FINAL_PLOTS=False,
    SAVE_FIGURES=False,
    return_fg=False,
    # job_name="fg_cfd_genbp_dev",
    # langevin parameters
    langevin_step_size=0.001,
    langevin_num_samples=5000,
    langevin_burn_in=1000,
    langevin_thinning=10,
    q_ylim=(-3,6),
)


executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=119,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_array_parallelism=50,
    slurm_mem=32*1024,
)
x = 2**np.arange(2,9)
# sweep_values = np.geomspace(0.1, 100, num=10)  # Exponential sweep for 'a'
# sweep_values = np.arange(16, 512, 32)
sweep_values = x.astype(int)

n_replicates = 10

exps = []
exp_names = []
jobinfo_paths = []

for method in [
        'gbp',
        'genbp',
        # 'laplace',
        'langevin'
        ]:
    experiment_name = f"{exp_prefix}_{method}_{sweep_param}"
    job_info = jobs2.submit_jobs(
        executor,
        run_run,
        {
            **base_kwargs,
            'method': method,
            'job_name': experiment_name,
        },
        sweep_param, sweep_values,
        n_replicates=n_replicates,
        experiment_name=experiment_name,
        batch=True
    )
    # Use helper functions to construct the file path and save the job info
    file_path = jobs2.construct_intermediate_path(f"{experiment_name}_jobinfo")
    jobs2.save_artefact(job_info, file_path)

    exps.append(job_info)
    exp_names.append(experiment_name)
    jobinfo_paths.append(file_path)

(
    gbp_experiment,
    genbp_experiment,
    # laplace_experiment,
    langevin
)= exps
(
    gbp_experiment_name,
    genbp_experiment_name,
    # laplace_experiment_name,
    langevin_experiment_name
)= exp_names
(
    gbp_jobinfo_path,
    genbp_jobinfo_path,
    # laplace_jobinfo_path,
    langevin_jobinfo_path
 ) = jobinfo_paths
print(f"gbp_experiment_name={gbp_experiment_name!r}")
print(f"genbp_experiment_name={genbp_experiment_name!r}")
# print(f"laplace_experiment_name={laplace_experiment_name!r}")
print(f"langevin_experiment_name={langevin_experiment_name!r}")
print(f"gbp_jobinfo_path={gbp_jobinfo_path!r}")
print(f"genbp_jobinfo_path={genbp_jobinfo_path!r}")
# print(f"laplace_jobinfo_path={laplace_jobinfo_path!r}")
print(f"langevin_jobinfo_path={langevin_jobinfo_path!r}")

#%% resume experiments
gbp_experiment_name=f'{exp_prefix}_gbp_{sweep_param}'
genbp_experiment_name=f'{exp_prefix}_genbp_{sweep_param}'
laplace_experiment_name=f'{exp_prefix}_laplace_{sweep_param}'
langevin_experiment_name=f'{exp_prefix}_langevin_{sweep_param}'
gbp_jobinfo_path=f'_logs/{exp_prefix}_gbp_{sweep_param}_jobinfo.pkl.bz2'
genbp_jobinfo_path=f'_logs/{exp_prefix}_genbp_{sweep_param}_jobinfo.pkl.bz2'
laplace_jobinfo_path=f'_logs/{exp_prefix}_laplace_{sweep_param}_jobinfo.pkl.bz2'
langevin_jobinfo_path=f'_logs/{exp_prefix}_langevin_{sweep_param}_jobinfo.pkl.bz2'
gbp_experiment = jobs2.load_artefact(gbp_jobinfo_path)
genbp_experiment = jobs2.load_artefact(genbp_jobinfo_path)
laplace_experiment = jobs2.load_artefact(laplace_jobinfo_path)
langevin_experiment = jobs2.load_artefact(langevin_jobinfo_path)

#%%
gbp_experiment_results = jobs2.collate_job_results(
    gbp_experiment,
    sweep_param
)
pprint(gbp_experiment_results)

jobs2.save_artefact(gbp_experiment_results, jobs2.construct_output_path(f"{gbp_experiment_name}"))

# %%
genbp_experiment_results = jobs2.collate_job_results(
    genbp_experiment,
    sweep_param
)
pprint(genbp_experiment_results)
jobs2.save_artefact(genbp_experiment_results, jobs2.construct_output_path(f"{genbp_experiment_name}"))

# %%
# laplace_experiment_results = jobs2.collate_job_results(
#     laplace_experiment,
#     sweep_param
# )
# pprint(laplace_experiment_results)
# jobs2.save_artefact(laplace_experiment_results, jobs2.construct_output_path(f"{laplace_experiment_name}"))
#%%
langevin_experiment_results = jobs2.collate_job_results(
    langevin_experiment,
    sweep_param
)
pprint(langevin_experiment_results)
jobs2.save_artefact(langevin_experiment_results, jobs2.construct_output_path(f"{langevin_experiment_name}"))

# %%
gbp_experiment_name=f'{exp_prefix}_gbp_{sweep_param}'
genbp_experiment_name=f'{exp_prefix}_genbp_{sweep_param}'
# laplace_experiment_name=f'{exp_prefix}_laplace_{sweep_param}'
gbp_jobinfo_path=f'_logs/{exp_prefix}_gbp_{sweep_param}_jobinfo.pkl.bz2'
genbp_jobinfo_path=f'_logs/{exp_prefix}_genbp_{sweep_param}_jobinfo.pkl.bz2'
# laplace_jobinfo_path=f'_logs/{exp_prefix}_laplace_{sweep_param}_jobinfo.pkl.bz2'
langevin_jobinfo_path=f'_logs/{exp_prefix}_langevin_{sweep_param}_jobinfo.pkl.bz2'
gbp_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{gbp_experiment_name}"))
genbp_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{genbp_experiment_name}"))
# laplace_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{laplace_experiment_name}"))
langevin_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{langevin_experiment_name}"))

y_keys = [
    'time',
    # 'memory',
    'q_mse',
    'q_loglik'
]
titles = [
    'Execution Time (s)',
    # 'Memory Usage',
    r'MSE',
    r'Log Likelihood'
]

better_high_dict = {
    'time': False,
    'q_mse': False,
    'q_loglik': True,
}

mode = "col"

# Now we create a plot of the performance of each method.
# We create several axes  to reuse to plot both experiments for time, memory, MSE and likelihood.
from tueplots import bundles, figsizes
n_plots = len(y_keys)

plt.rcParams.update(bundles.iclr2024())
plt.rcParams['text.latex.preamble'] = plt.rcParams['text.latex.preamble'] + r'\usepackage{mathrsfs}'
if mode == "col":
    plt.rcParams.update(figsizes.iclr2024(nrows=n_plots, ncols=1))
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)
else:
    plt.rcParams.update(figsizes.iclr2024(ncols=n_plots, nrows=1, height_to_width_ratio=1.0))
    fig, axs = plt.subplots(ncols=n_plots, nrows=1)

# Determine the maximum x-value from the genbp experiment results to set a consistent right limit for the x-axis
max_genbp_x = max([d**2 for d in genbp_experiment_results.keys()])

for i, ax in enumerate(axs):
    y_key = y_keys[i]
    title = titles[i]
    better_high = better_high_dict[y_key]
    # x axis should be log scale
    ax.set_xscale('log')
    gbp_ds = []
    gbp_vals = []

    for d in gbp_experiment_results.keys():
        d2 = d**2
        gbp_ds.append(d2)
        quantiles = np.nanpercentile(gbp_experiment_results[d][y_key],
            [0, 25, 50, 75, 100])
        if y_key =='q_loglik':
            quantiles /= d2
        gbp_vals.append(quantiles)
    gbp_ds = np.array(gbp_ds)
    gbp_vals = np.array(gbp_vals)
    # plot quantiles of gbp
    medians = gbp_vals[:,2]
    lower = gbp_vals[:,1]
    upper = gbp_vals[:,3]
    lower_err = medians - lower
    upper_err = upper - medians
    ax.errorbar(
        gbp_ds, medians, yerr=[lower_err, upper_err],
        fmt='_', label='GaBP', color='blue'
    )

    # Plot GEnBP results
    genbp_ds = []
    genbp_vals = []
    for d in genbp_experiment_results.keys():
        d2 = d**2
        genbp_ds.append(d2)
        quantiles = np.nanpercentile(genbp_experiment_results[d][y_key],
            [0, 25, 50, 75, 100])
        if y_key =='q_loglik':
            quantiles /= d2
        genbp_vals.append(quantiles)

    genbp_ds = np.array(genbp_ds)
    genbp_vals = np.array(genbp_vals)
    # plot quantiles of genbp
    medians = genbp_vals[:,2]
    lower = genbp_vals[:,1]
    upper = genbp_vals[:,3]
    lower_err = medians - lower
    upper_err = upper - medians
    ax.errorbar(
        genbp_ds, medians, yerr=[lower_err, upper_err],
        fmt='_', label='GEnBP', color='green'
    )

    # Plot Laplace results
    laplace_ds = []
    laplace_vals = []
    for d in laplace_experiment_results.keys():
        d2 = d**2
        laplace_ds.append(d2)
        quantiles = np.nanpercentile(laplace_experiment_results[d][y_key],
                                     [0, 25, 50, 75, 100])
        if y_key == 'q_loglik':
            quantiles /= d2
        laplace_vals.append(quantiles)

    laplace_ds = np.array(laplace_ds)
    laplace_vals = np.array(laplace_vals)
    # plot quantiles of laplace
    medians = laplace_vals[:, 2]
    lower = laplace_vals[:, 1]
    upper = laplace_vals[:, 3]
    lower_err = medians - lower
    upper_err = upper - medians
    ax.errorbar(
        laplace_ds, medians, yerr=[lower_err, upper_err],
        fmt='_', label='Laplace', color='brown'
    )

    # Plot Langevin results
    langevin_ds = []
    langevin_vals = []
    for d in langevin_experiment_results.keys():
        d2 = d**2
        langevin_ds.append(d2)
        quantiles = np.nanpercentile(langevin_experiment_results[d][y_key],
                                     [0, 25, 50, 75, 100])
        if y_key == 'q_loglik':
            quantiles /= d2
        langevin_vals.append(quantiles)

    if y_key in ("time",):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0e}'.format(y)))
    else:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    if i==0:
        ax.legend(fontsize=12)

    if (i == len(axs) - 1):
        ax.set_xlabel(r"$D_\mathscr{Q}$")
    ax.set_ylabel(title)

    # Set the x-axis limits explicitly based on the genbp values
    ax.set_xlim(left=gbp_ds.min(), right=max_genbp_x)
    gbp_last_x = gbp_ds[-1]

    # Shade the area to the right of the last GaBP point
    ax.axvspan(gbp_last_x, max_genbp_x, color='grey', alpha=0.5, label='GaBP OOM' )

    # Set arrow properties
    # arrow_color = 'green' if better_high else 'red'
    arrow_color = 'black'
    arrow_direction = '<-' if better_high else '->'

    # Place the arrow outside the plot in the left margin
    # `transform=ax.transAxes` means position is relative to the axes boundary, not the data
    # Calculate starting and ending points for the arrow
    base_y = 0.05
    head_y = 0.15
    # if better_high:
    #     base_y, head_y = head_y, base_y

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



    if i == 0:  # Add the label only once to avoid duplicate legend entries
        ax.legend()


# plt.tight_layout()
# save figure as PDF in FIG_DIR
plt.savefig(
    os.path.join(FIG_DIR, f'fg_cfd_{sweep_param}_sweep.pdf'), bbox_inches='tight')
plt.savefig(
    os.path.join(FIG_DIR, f'fg_cfd_{sweep_param}_sweep.png'), bbox_inches='tight')
plt.show()

# %%
