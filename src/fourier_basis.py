"""
This file provides a fourier basis implemented manually;
Going the obvious way, via the FFT, breaks some numpyro functions because complex-valued functions do not work RN.
So we do it a slower way, and gain some flexibility.
"""

from math import sqrt, ceil, floor

import torch
import torch.nn as nn
from einops import rearrange, repeat, asnumpy
from functools import reduce
from operator import mul


def phase(
        dim_sizes,
        pos_low: float=-1,
        pos_high:float=1,
        centered: bool=True,
        dtype=None):
    """
    Basically a clone of `encode_positions`,
    but reimplemented here to keep the other thing unpickling correctly.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    def generate_grid(size):
        width = (pos_high - pos_low) / size
        if centered:
            left = pos_low + width/2
            right = pos_high - width/2
        else:
            left = pos_low
            right = pos_high - width
        return torch.linspace(
            left, right, steps=size,
            dtype=dtype)
    grid_list = list(map(generate_grid, dim_sizes))
    grid = torch.stack(torch.meshgrid(*grid_list), dim=-1)
    # # stack never fails to increase the number of dimensions, which is not what we want for univariate grids
    # if len(dim_sizes) == 1:
    #     grid = grid.squeeze(-1)
    return grid

def basis(
        grid, fs_even, fs_odd=None,
        norm=True # Normalise to be "unitary" in the sense that the inner product is 1
    ):
    """
    map a list of frequency tuples over a list of grids,
    return cos and sin bases using these frequencies.
    Not an efficient way of handling DC.
    untested for d>2.

    Example:
    >>> grid = phase((10, 10))
    >>> fs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    >>> basis(grid, *fs)
    """
    if fs_odd is None:
        #really this codepath is for convenience of testing only.
        fs_odd = fs_even
    #hack because it feels weird to return fs with a dim index in 1d
    if fs_even.ndim == 1:
        fs_even = fs_even.unsqueeze(-1)
    if fs_odd.ndim == 1:
        fs_odd = fs_odd.unsqueeze(-1)
    phases_even = torch.einsum("...i,ki->k...", grid, fs_even) * torch.pi
    phases_odd = torch.einsum("...i,ki->k...", grid, fs_odd) * torch.pi
    # Calling trig functions profligately assuming that this will be cached
    # basis = torch.cat([torch.cos(phases_even), torch.sin(phases_odd)], dim=0)
    basis_even = torch.cos(phases_even)
    basis_odd = torch.sin(phases_odd)
    scale = reduce(mul, grid.shape[:-1])  #last axis is packed with dims
    # print("scale", grid.shape[:-1], sqrt(scale))
    if norm:
        basis_even /= (sqrt(scale/2))
        basis_odd /= (sqrt(scale/2))
    return basis_even, basis_odd


def basis_flat(
        *args, **kwargs
    ):
    """
    same as `basis` but interleaves sine and cos terms.

    Example:
    >>> grid = phase((10, 10))
    >>> fs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    >>> basis_flat(grid, *fs)
    """
    return torch.cat(basis(*args, **kwargs), dim=0)


def complete_fs(
        dim_ranges, dc=False,
        dtype=None, sorted=False):
    """
    Generate a complete list of frequency tuples for given set of spectral ranges, excluding linear combinations.
    User is responsible for avoiding aliasing.
    Negative frequencies in the first dimension are ignored.

    Example:
    >>> phases = fourier_basis.phase((16,16))
    >>> fs = fourier_basis.complete_fs((3,3), dc=True)
    >>> bases = fourier_basis.basis(phases, *fs, norm=False)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    ## When the first axis is 0 we have some extra symmetries
    # I THINK this handles those; have not thoroughly tested in d>2
    dim_ranges = list(dim_ranges)
    # We allow dim_ranges to be tuples so that we can be asymmetric;
    # this only makes sense for (low, high) pairs if abs(low+high) <=1
    for i,r in enumerate(dim_ranges):
        if isinstance(r, (float, int)):
            if i > 0:
                dim_ranges[i] = (-r, r)
            else:
                # First axis is special because of symmetry of real transform
                dim_ranges[i] = (0, r)

    # range_even = [torch.zeros((1,), dtype=dtype)]
    # range_odd = [torch.arange(1, dim_ranges[0][1]+1, dtype=dtype)]
    range_even = [torch.arange(0, dim_ranges[0][1]+1, dtype=dtype)]
    range_odd = [torch.arange(1, dim_ranges[0][1]+1, dtype=dtype)]
    for l, h in dim_ranges[1:]:
        range_even.append(
            torch.arange(0, h+1, dtype=dtype))
        range_odd.append(
            torch.arange(l, h+1, dtype=dtype))
    # print("range_even", range_even)
    # print("range_odd", range_odd)

    fs_even = torch.cartesian_prod(*range_even)
    fs_odd = torch.cartesian_prod(*range_odd)
    if not dc:
        fs_even = fs_even[1:]

    # we usually introduce the bias term elsewhere and anyway sin(0)=0
    # sort by frequency
    if sorted:
        fs_even = fs_even[torch.linalg.norm(fs_even,dim=1).argsort()]
        fs_odd = fs_odd[torch.linalg.norm(fs_odd,dim=1).argsort()]
    return (fs_even.contiguous(), fs_odd.contiguous())
