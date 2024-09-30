
# %%
"""Performance in high dimension
"""
# %load_ext autoreload
# %autoreload 2
%run mp_030_factor_graph_sysid_ns2d_wrapped.py

from src import jobs2
from src.jobs import *
exp_prefix = "fg_cfd_dev"
sweep_param = 'visc'

sweep_values = np.geomspace(1e-4, 1e4, num=19, endpoint=True)  # Exponential sweep for 'a'
# sweep_values = np.arange(16, 512, 32)
# x = 10**np.arange(-4,4, dtype=float)
# sweep_values = x.astype(int)

n_replicates = 10

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


exps = []
exp_names = []
jobinfo_paths = []

for method in ['gbp', 'genbp']:
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

gbp_experiment, genbp_experiment = exps
gbp_experiment_name, genbp_experiment_name = exp_names
gbp_jobinfo_path, genbp_jobinfo_path = jobinfo_paths
print(f"gbp_experiment_name={gbp_experiment_name!r}")
print(f"genbp_experiment_name={genbp_experiment_name!r}")
print(f"gbp_jobinfo_path={gbp_jobinfo_path!r}")
print(f"genbp_jobinfo_path={genbp_jobinfo_path!r}")

#%% resume experiments
gbp_experiment_name=f'{exp_prefix}_gbp_{sweep_param}'
genbp_experiment_name=f'{exp_prefix}_genbp_{sweep_param}'
gbp_jobinfo_path=f'_logs/{exp_prefix}_gbp_{sweep_param}_jobinfo.pkl.bz2'
genbp_jobinfo_path=f'_logs/{exp_prefix}_genbp_{sweep_param}_jobinfo.pkl.bz2'
gbp_experiment = jobs2.load_artefact(gbp_jobinfo_path)
genbp_experiment = jobs2.load_artefact(genbp_jobinfo_path)
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
gbp_experiment_name=f'{exp_prefix}_gbp_{sweep_param}'
genbp_experiment_name=f'{exp_prefix}_genbp_{sweep_param}'
gbp_jobinfo_path=f'_logs/{exp_prefix}_gbp_{sweep_param}_jobinfo.pkl.bz2'
genbp_jobinfo_path=f'_logs/{exp_prefix}_genbp_{sweep_param}_jobinfo.pkl.bz2'
gbp_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{gbp_experiment_name}"))
genbp_experiment_results = jobs2.load_artefact(jobs2.construct_output_path(f"{genbp_experiment_name}"))

y_keys = [
    'time',
    # 'memory',
    'q_mse',
    'q_loglik'
]
titles = [
    'Execution Time (s)',
    # 'Memory Usage',
    r'Mean-Squared Error',
    r'Log Likelihood'
]


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
    plt.rcParams.update(figsizes.iclr2024(ncols=n_plots, nrows=1))
    fig, axs = plt.subplots(ncols=n_plots, nrows=1)


fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)

for i, ax in enumerate(axs):
    y_key = y_keys[i]
    title = titles[i]
    ax.set_xscale('log')
    gbp_ds = []
    gbp_vals = []
    for d in gbp_experiment_results.keys():
        gbp_ds.append(d**2)
        gbp_vals.append(
            np.nanpercentile(gbp_experiment_results[d][y_key],
            [0, 25, 50, 75, 100]))
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
    # x axis should be log scale

    genbp_ds = []
    genbp_vals = []
    for d in genbp_experiment_results.keys():
        genbp_ds.append(d**2)
        genbp_vals.append(
            np.nanpercentile(genbp_experiment_results[d][y_key],
            [0, 25, 50, 75, 100]))
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
    # x axis should be log scale


    if i==0:
        ax.legend(fontsize=12)

    if (i == len(axs) - 1):
        ax.set_xlabel(r"$D_\mathscr{Q}$")
    ax.set_ylabel(title)

    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Add title inside the plot area at the bottom left corner
    # ax.text(
    #     0.05, 0.05, title, transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
    #     bbox=dict(
    #         facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
    #     )

    # ax.text(
    #     0.95, 0.95, title, transform=ax.transAxes, fontsize=10, va='top', ha='right',
    #     bbox=dict(
    #         facecolor='white', alpha=0.7,
    #         edgecolor='none', boxstyle='round,pad=0.5'))

# plt.tight_layout()
# save figure as PDF in FIG_DIR
plt.savefig(
    os.path.join(FIG_DIR, f'fg_cfd_{sweep_param}_sweep.pdf'), bbox_inches='tight')
plt.show()

# %%
