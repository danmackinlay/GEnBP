"""
Adapted from code by Zongyi Li
included under MIT license
"""
import math
from enum import Enum

import numpy as np
import torch
from einops import rearrange, repeat
from .math_helpers import convert_1d_to_2d, convert_2d_to_1d

"""
This navier-stokes sim stores data as (batch, x, y,) i.e. batch first
"""

# w0: initial vorticity
# f: forcing term
#visc: viscosity (1/Re)
# delta_t: internal time-step for solve (descrease if blow-up)
# interval: number of internal time-steps between observations
def navier_stokes_2d_step(
        w0, f, visc=1.0, delta_t=0.01, interval=1, v_noise_power=0.):

    # Grid size - must be power of 2
    N = w0.shape[-1]
    noise_power_per_cell = v_noise_power / (w0.shape[-2] * w0.shape[-1])

    # Maximum frequency
    k_max = math.floor(N / 2)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[-2, -1], norm='backward')

    # Forcing to Fourier space
    f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')

    # If same forcing for the whole batch
    if len(f_h.shape) < len(w_h.shape):
        f_h = rearrange(f_h, '... -> 1 ...')

    # Wavenumbers in y-direction
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=w0.device),
        torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
        0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0

    # Dealiasing mask
    dealias = torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max
        ).float()
    if len(dealias.shape) < len(w_h.shape):
        dealias = torch.unsqueeze(dealias, 0)

    for j in range(interval):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        q_real_temp = q.real.clone()
        q.real = -2 * math.pi * k_y * q.imag
        q.imag = 2 * math.pi * k_y * q_real_temp
        q = torch.fft.ifftn(q, dim=[-2, -1], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        v_real_temp = v.real.clone()
        v.real = 2 * math.pi * k_x * v.imag
        v.imag = -2 * math.pi * k_x * v_real_temp
        v = torch.fft.ifftn(v, dim=[-2, -1], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        w_x_temp = w_x.real.clone()
        w_x.real = -2 * math.pi * k_x * w_x.imag
        w_x.imag = 2 * math.pi * k_x * w_x_temp
        w_x = torch.fft.ifftn(w_x, dim=[-2, -1], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        w_y_temp = w_y.real.clone()
        w_y.real = -2 * math.pi * k_y * w_y.imag
        w_y.imag = 2 * math.pi * k_y * w_y_temp
        w_y = torch.fft.ifftn(w_y, dim=[-2, -1], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(
            q * w_x + v * w_y,
            dim=[-2, -1], norm='backward')

        # Dealias
        F_h *= dealias

        # noisy dynamics
        eta = torch.randn(
            w0.shape, dtype=w0.dtype, device=w0.device) * noise_power_per_cell**0.5
        # noise to Fourier space
        # this is a lazy way to do it; we could do the noise in Fourier space
        eta_h = torch.fft.fftn(eta, dim=[-2, -1], norm='backward')

        # Cranck-Nicholson update
        factor = 0.5 * delta_t * visc * lap
        num = (
            -delta_t * F_h
            + delta_t * f_h
            + delta_t * eta_h
            + (1.0 - factor) * w_h
        )
        w_h = num / (1.0 + factor)

    w = torch.fft.ifftn(
        w_h, dim=[-2, -1], norm='backward').real
    if w.isnan().any().item():
        raise ValueError(w.isnan().sum(), 'NaN values found.')
    return w


def navier_stokes_2d_step_vector_form(w0, f, visc=1.0, delta_t=0.01, interval=1, v_noise_power=0.):
    # original_shape = w0.shape
    # grid_size = int(original_shape[-1] ** 0.5)  # Assuming w0 is a square 2D array or batch of square 2D arrays

    assert len(w0.shape) == 2, 'w0 must be a batch of vectors'
    assert len(f.shape) == 2, 'f must be a batch of vectors'
    # Convert w0 from 1D to 2D
    w0_reshaped = convert_1d_to_2d(w0)
    f_reshaped = convert_1d_to_2d(f)

    # Call the original function
    result = navier_stokes_2d_step(w0_reshaped, f_reshaped, visc, delta_t, interval, v_noise_power)

    # Reshape the result back to its original shape
    return convert_2d_to_1d(result)
