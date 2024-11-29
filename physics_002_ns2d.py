# %%
# Navier stokes solving using our custom sim

%load_ext autoreload
%autoreload 2

import os
from dotenv import load_dotenv

import numpy as np
from math import sqrt, ceil, floor

from einops import asnumpy, rearrange
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tueplots import bundles, figsizes
plt.rcParams.update(bundles.iclr2024())
plt.rcParams['text.latex.preamble'] = plt.rcParams['text.latex.preamble'] + r'\usepackage{mathrsfs}'

import numpy as np

torch.set_default_dtype(torch.float64)
import lovely_tensors as lt
lt.monkey_patch()

from src import ns_2d, random_fields

TORCH_DEVICE = 'cpu'

#%%
n_steps = 5
d = 64
# visc = 0.1
delta_t = 0.01
interval = 20
v_noise_power = 1e2

# plt.rcParams.update(figsizes.iclr2024(ncols=n_steps, nrows=2))

# def observation_operator(image, downsample_factor, noise_std):
#     """
#     Observation by downsampling and adding noise
#     """

#     downsampled_image = image[::downsample_factor, ::downsample_factor]
#     noisy_image = downsampled_image + np.random.normal(
#         scale=noise_std, size=downsampled_image.shape)
#     return noisy_image

grf = random_fields.GaussianRF(2, d, alpha=3.5, tau=10)
f = grf.sample(1)
# plt.imshow(f[0], cmap='cool')
# plt.title('Forcing term q')
# plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), d-0.5, d-0.5, linewidth=2, edgecolor='black', linestyle='dotted', fill=False))
# plt.savefig('fig/physics_002_ns2d_f.png', transparent=True)
# plt.show()
w0 = random_fields.GaussianRF(2, d, alpha=2.0, tau=8).sample(1)

corrupt_scale = torch.std(w0) * 0.1

for visc_exp in (-4., -1., 2.):
    fig, axs = plt.subplots(ncols=n_steps, nrows=2)
    visc = 10.**visc_exp
    w = w0
    for t in range(n_steps):
        if t == 0:
            axs[1, t].imshow(f[0], cmap='cool')
            axs[1, t].add_patch(
                plt.Rectangle(
                    (-0.5, -0.5), d-0.5, d-0.5, linewidth=2, edgecolor='black', linestyle='dotted', fill=False))
            axs[1, t].text(-0.1, 0.5, '$\mathsf{q}$', transform=axs[1, t].transAxes, ha='right', va='center')
        else:
            print(f"t={t}")
            w = ns_2d.navier_stokes_2d_step(
                w, f=f, visc=visc, delta_t=delta_t,
                interval=interval, v_noise_power=v_noise_power
            )
            y = observation_operator(w[0], downsample_factor=8, noise_std=corrupt_scale)
            axs[1, t].imshow(y, interpolation='none')
            axs[1, t].text(-0.1, 0.5, r'$\mathsf{y}_' + f'{t}$', transform=axs[1, t].transAxes, ha='right', va='center')
        axs[1, t].axis('off')
        axs[0, t].imshow(w[0])
        axs[0, t].text(-0.1, 0.5, r'$\mathsf{x}_' + f'{t}$', transform=axs[0, t].transAxes, ha='right', va='center')
        axs[0, t].axis('off')

    fig.savefig(f'fig/physics_002_ns2d_{int(visc_exp)}.png', transparent=True)
    fig.savefig(f'fig/physics_002_ns2d_{int(visc_exp)}.pdf', transparent=True)
    fig.show()
# %%


# %%
# batched vector version
from src.ns_2d import convert_1d_to_2d, convert_2d_to_1d, navier_stokes_2d_step_vector_form, navier_stokes_2d_step
from src.random_fields import GaussianRF
n_ens = 150

grf = GaussianRF(
    2, d, alpha=3.5, tau=5)
q_2d = grf.sample(n_ens)[0]
x0_2d = GaussianRF(2, d).sample(1)[0]
q_prior_ens_2d = grf.sample(n_ens)
q = convert_2d_to_1d(q_2d)
x0 = convert_2d_to_1d(x0_2d)
q_prior_ens = convert_2d_to_1d(q_prior_ens_2d)

#%%

d = 4
visc = 0.5
delta_t = 0.01
interval = 2
v_noise_power = 1e2

grf_w0 = GaussianRF(
    2, d, alpha=2.5, tau=3)

grf_f = GaussianRF(
    2, d, alpha=3.5, tau=0)


def compute_covariance(grf_w0, grf_f, n_ens):
    # Step 1: Generate batches
    w0_batch = grf_w0.sample(n_ens)
    f_batch = grf_f.sample(n_ens)

    # Step 2: Compute w1 using the navier_stokes_2d_step_vector_form
    w1_batch = navier_stokes_2d_step(w0_batch, f_batch)

    # Step 3: Flatten w0, f, and w1
    # w0_flat = convert_2d_to_1d(w0_batch)
    # f_flat = convert_2d_to_1d(f_batch)
    # w1_flat = convert_2d_to_1d(w1_batch)
    w0_batch = convert_2d_to_1d(w0_batch)
    f_batch = convert_2d_to_1d(f_batch)
    w1_batch = convert_2d_to_1d(w1_batch)

    # Step 4: Stack the flattened arrays
    data = torch.cat((w0_batch, f_batch, w1_batch), dim=1)

    # Step 5: Compute the covariance matrix
    data_mean = torch.mean(data, dim=0)
    data_centered = data - data_mean
    covariance_matrix = torch.matmul(data_centered.T, data_centered) / (n_ens - 1)

    return covariance_matrix.numpy()


def visualize_covariance(covariance_matrix):
    # Step 6: Visualize the covariance matrix
    plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Covariance Matrix Visualization")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.show()


# Example usage
n_ens = 64  # Sample size
covariance_matrix = compute_covariance(grf_w0, grf_f, n_ens)
visualize_covariance(covariance_matrix)
plt.show()

# %%
