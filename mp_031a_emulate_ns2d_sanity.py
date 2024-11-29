# %%
"""Baseline model inference functionality
"""
import torch
from src.torch_formatting import (
    flatten_model_parameters,
    unflatten_model_parameters_batch,
    set_model_parameters,
    flatten_tensor_batch,
    unflatten_tensor_batch,
    batched_vector_model,
    normalize_model_dtypes,
    flatten_gradient_variances,
)
from mp_031_emulate_ns2d_wrapped import *

dtype = torch.float32
torch.set_default_dtype(dtype)
n_ens = 30
job_name = "nsemu_sanity"

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
    # alpha_scale=2.0,
)

# You can use submitit for job submission if desired
# executor = submitit.AutoExecutor(folder=LOG_DIR)
# executor = submitit.DebugExecutor(folder=LOG_DIR)
# executor.update_parameters(
#     timeout_min=199,
#     # gpus_per_node=1,
#     slurm_account=os.getenv('SLURM_ACCOUNT'),
#     slurm_mem=64*1024,
# )
# j = executor.submit(run_run, **base_kwargs, job_name=f"cfd_{method}_forney_sanity", seed=1)
# result = j.result()

# Direct invocation of the run_run function
result, reproduction = run_run(
    **base_kwargs,
    job_name=job_name,
    seed=74
)

# Print the results
from pprint import pprint
pprint(result)
# %%
# Out of sample testing. We generate a new dataset and test the model on it, rolling out the model in time using the original and new emulators.
def process_run():
    # Direct invocation of the run_run function
result, reproduction = run_run(
    **base_kwargs,
    job_name=job_name,
    seed=74
)
import torch
from src.torch_formatting import (
    flatten_model_parameters,
    unflatten_model_parameters_batch,
    set_model_parameters,
    flatten_tensor_batch,
    unflatten_tensor_batch,
    batched_vector_model,
    normalize_model_dtypes,
    flatten_gradient_variances,
)
from mp_031_emulate_ns2d_wrapped import *
from matplotlib import pyplot as plt

mmd_scale = 20.
rep_path = f"{OUTPUT_DIR}/_{job_name}_res.pkl.gz"
with gzip.open(rep_path, "rb") as f:
    reproduction = cloudpickle.load(f)
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

# Plot the MMD over time
plt.figure(figsize=(10, 6))
plt.plot(range(timesteps), mmd_original, label='Original Emulator MMD')
plt.plot(range(timesteps), mmd_new, label='Adapted Emulator MMD')
plt.xlabel('Time Step')
plt.ylabel('MMD')
plt.title('MMD over Time Compared to Ground Truth')
plt.legend()
plt.savefig(f"{FIG_DIR}/{job_name}_mmd.png")
plt.savefig(f"{FIG_DIR}/{job_name}_mmd.pdf")
plt.show()

# %%