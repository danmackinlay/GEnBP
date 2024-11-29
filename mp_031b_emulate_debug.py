# %%
"""Let's test convergence of an adapted emulator
"""
# %load_ext autoreload
# %autoreload 2
from src.jobs import *
# %run mp_030_factor_graph_sysid_ns2d_wrapped.py
from mp_031_emulate_ns2d_wrapped import *

job_name = "nsemu_erd"
dtype = torch.float32
torch.set_default_dtype(dtype)
n_ens = 30

#%% sweep some params
base_kwargs = dict(
    d=32,
    visc=0.001,
    delta_t=0.01,
    interval=10,
    v_noise_power=1e2,
    downsample=5,
    obs_sigma2=0.01,  # Observation noise variance
    n_timesteps=5,
    latent_scale=4.0,
    ## Inference parameters
    n_ens=n_ens,
    cvg_tol=-1.0,
    damping=0.0,  # no damping
    hard_damping=True,
    inf_gamma2=0.1,  # BP noise
    q_gamma2=0.001,   # BP noise
    w_gamma2=0.1,    # BP noise
    inf_sigma2=0.1,  # Inference noise
    q_sigma2=0.01,   # Inference noise
    w_sigma2=0.01,    # Inference noise
    inf_eta2=0.01,    # Conformation noise
    w_inflation=1000.0,  # diffuse weight prior
    max_rank=n_ens,
    max_steps=150,
    rtol=1e-6,
    atol=1e-8,
    empty_inboxes=True,
    min_mp_steps=3,
    max_relin_int=5,
    belief_retain_all=False,
    conform_retain_all=True,
    conform_r_eigen_floor=1e-4,
    conform_randomize=True,
    ## Diagnostics
    DEBUG_MODE=False,
    DEBUG_PLOTS=False,
    FINAL_PLOTS=True,
    SAVE_FIGURES=True,
    SHOW_FIGURES=True,
    return_fg=True,
    fno_n_modes=12,
    fno_hidden_channels=32,
    fno_n_layers=4,
    fno_typ="FNO",
    cheat=0.0,  # when >0, bias towards the truth for testing
    # q_ylim=(-1, 1),
    lw=0.2,
    lateness=1.0,
    sparsify_alpha=0.2,
    mmd_scale = 20.,
    # alpha_scale=2.0,
)

def process_run(seed=0, mmd_scale=10, **kwargs,):
    # Direct invocation of the run_run function
    result, reproduction = run_run(
        **kwargs,
        job_name=job_name,
        seed=seed
    )
    print("Finished run")
    # rep_path = f"{OUTPUT_DIR}/_{job_name}_res.pkl.gz"
    # with gzip.open(rep_path, "rb") as f:
    #     reproduction = cloudpickle.load(f)
    # roll out the model in time using the original emulator,  the new emulator, and also the truth
    fg_real = reproduction['fg_real']
    fg = reproduction['fg']
    # x0_2d = reproduction['x_grf'].sample(30)
    # q, q_shape = flatten_tensor_batch(q_2d.unsqueeze(0))
    # x0, x_shape = flatten_tensor_batch(x0_2d.unsqueeze(0))
    x0 = fg.get_var_node('x5').get_ens()
    q_node = fg_real.get_var_node('q')
    sim_wxhat_xhatp = reproduction['sim_wxhat_xhatp']
    sim_qx_xp = reproduction['sim_qx_xp']

    original_w = reproduction['w_prior_ens']
    original_q = fg_real.get_var_node('q').get_ens()
    posterior_w = fg.get_var_node('w').get_ens()

    # Define the Gaussian kernel function
    def gaussian_kernel(x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    # Define the MMD computation function
    def compute_mmd(x, y, sigma=0.1):
        xx = gaussian_kernel(x, x, sigma)
        yy = gaussian_kernel(y, y, sigma)
        xy = gaussian_kernel(x, y, sigma)
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd.item()

    x_states_truth = [x0]
    x_states_original_emulator = [x0]
    x_states_new_emulator = [x0]

    for t in range(1, 5):
        #roll out truth and 2 emulators on new data
        [x_truth] = sim_qx_xp(
            original_q, x_states_truth[-1])
        x_states_truth.append(x_truth)

        [x_original_emulator] = sim_wxhat_xhatp(original_w, x_states_original_emulator[-1])
        x_states_original_emulator.append(x_original_emulator)

        [x_new_emulator] = sim_wxhat_xhatp(posterior_w, x_states_new_emulator[-1])
        x_states_new_emulator.append(x_new_emulator)

    # plot time series showing the evolution of error in the state
    # Compute RMSE over time
    timesteps = len(x_states_truth)
    mmd_original = []
    mmd_new = []

    for t in range(timesteps):
        # Flatten the tensors to (n_samples, n_features)
        x_truth_flat, _ = flatten_tensor_batch(x_states_truth[t])
        x_original_flat, _ = flatten_tensor_batch(x_states_original_emulator[t])
        x_new_flat, _ = flatten_tensor_batch(x_states_new_emulator[t])

        # Compute MMD between the original emulator and the truth
        mmd_orig = compute_mmd(x_original_flat, x_truth_flat, sigma=mmd_scale)
        mmd_original.append(mmd_orig)

        # Compute MMD between the adapted emulator and the truth
        mmd_adapted = compute_mmd(x_new_flat, x_truth_flat, sigma=mmd_scale)
        mmd_new.append(mmd_adapted)
    mmds = {
        'mmd_original': mmd_original,
        'mmd_new': mmd_new,
        'timesteps': timesteps
    }
    cloudpickle.dump(mmds, open(f"{OUTPUT_DIR}/{job_name}_mmd.pkl", "wb"))
    return mmds


# executor = submitit.AutoExecutor(folder=LOG_DIR)
executor = submitit.DebugExecutor(folder=LOG_DIR)
executor.update_parameters(
    timeout_min=119,
    # gpus_per_node=1,
    slurm_account=os.getenv('SLURM_ACCOUNT'),
    slurm_array_parallelism=50,
    slurm_mem=64*1024,
)
executor.update_parameters(name=job_name)

n_replicates = 1
job_info = []

seed = 74
with executor.batch():
    for r in range(n_replicates):
        kwargs = base_kwargs.copy()
        kwargs['seed'] = seed
        seed += 1
        print(f"experiment_name: {job_name} seed={seed}")
        job = executor.submit(process_run, **kwargs)
        job_info.append({'job': job, 'params': kwargs})

with gzip.open(os.path.join(LOG_DIR, f'{job_name}_job_info.pkl.gz'), 'wb') as f:
    cloudpickle.dump(job_info, f)

# %%
job_results = []
for job_info in job_info:
    job = job_info['job']
    params = job_info['params']
    result = job.result()
    job_results.append({'job': job, 'params': params, 'result': result})

save_artefact(job_results, f"{job_name}_mmd", OUTPUT_DIR)
# %%
job_results = load_artefact(f"{job_name}_mmd", OUTPUT_DIR)

# Now we create a plot of the performance of each method.
# We create several axes  to reuse to plot both experiments for time, memory, MSE and likelihood.
from tueplots import bundles, figsizes
import matplotlib as mpl
plt.rcParams.update(bundles.iclr2024())
plt.rcParams['text.latex.preamble'] = plt.rcParams['text.latex.preamble'] + r'\usepackage{mathrsfs}'

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
    os.path.join(FIG_DIR, f'{job_name}_sweep.pdf'), bbox_inches='tight')
plt.show()

# %%
