from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from math import sqrt, ceil, floor
import numpy as np
from matplotlib.colors import Normalize
from einops import asnumpy
from .gaussian_statistics import fake_ens_from_moments, moments_from_ens


def multi_img_plot(x, interval=1, n_cols=None, fsize=6, interpolation=None, crange=None):
    """
    Plot array as images, iterating over first axis for each successive one.
    """
    x = asnumpy(x)
    steps = range(0, x.shape[0], interval)
    if n_cols is None:
        n_cols = len(steps)
    print(steps, len(steps), n_cols)
    n_rows = ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsize * n_cols/n_rows, fsize));
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    if crange is None:
        vmax = max([np.amax(x), -np.amin(x)])
        vmin = -vmax
    elif isinstance(crange, (int, float)):
        vmin = -crange
        vmax = crange
    else:  # tuple, presumably
        vmin, vmax = crange

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    for i, ax in zip(steps, axes):
        ax.imshow(asnumpy(x[i,...]), interpolation=interpolation, norm=norm, cmap=cmap)
    # fig.colorbar(ims[0], ax=axs, orientation='vertical', shrink = 0.6)
    return fig


def img_plot(x, interval=1, n_cols=None, fsize=6, interpolation=None):
    """
    plot whatever image I can find in x
    """
    x = np.squeeze(x, )
    plt.imshow(x, interpolation=interpolation)
    plt.gca().set_axis_off()
    plt.tight_layout()
    return plt.gcf()


def multi_heatmap(xs, names=None, crange=None, base_size=3.0, dpi=100):
    n_ims = len(xs)
    if names is None:
        names = range(n_ims)
    fig, axs = plt.subplots(
        1, n_ims, figsize=(base_size*n_ims, base_size),
        dpi=dpi)
    if crange is None:
        vmax = max([np.abs(x).max() for x in xs])
        vmin = -vmax
    else:
        vmin = -crange
        vmax = crange
    ims = []
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    if n_ims == 1:
        # ffs matplotlib "helps" by flattening the array. single subplot is not a thing.
        iter_axs = [axs]
    else:
        iter_axs = axs

    for imi, (x, name, ax) in enumerate(zip(xs, names, iter_axs)):
        ax.set_axis_off()
        ax.set_title(name)
        im = ax.imshow(x, interpolation='bilinear', norm=norm, cmap=cmap)
        ims.append(im)

    fig.colorbar(ims[0], ax=axs, orientation='vertical', shrink = 0.6)
    # fig.tight_layout()
    return fig


def pred_target_heatmap(pred, y, * args, **kwargs):
    return multi_heatmap(
        [pred, y, pred-y],
        names=['pred', 'target', 'diff'],
        *args, **kwargs)


def meshify(X, Z):
    """
    takes a 2D array X, containing pairs of grid coordinates, and an array Z
    containing values at those coordinates, and return a
    meshgrid-compatible representation of X, and corresponding Z values
    """
    X = np.asarray(X)
    Z = np.asarray(Z)
    sorter = np.lexsort((X[:,1], X[:,0]))
    X = X[sorter]
    Z = Z[sorter]
    n_x = np.unique(X[:,0]).shape[0]
    n_y = np.unique(X[:,1]).shape[0]

    # mesh_X = rearrange(
    #     X, "n (x y) -> x y n", x=n_x, y=n_y
    # )
    mesh_x = X[:, 0].reshape(n_x, n_y)
    mesh_y = X[:, 1].reshape(n_x, n_y)
    mesh_Z = Z.reshape(n_x, n_y)
    return mesh_x, mesh_y, mesh_Z


def inbox_plot(node, truth=None, trunc=None, offset=0, step=1):
    """
    plot my incoming messages
    """
    from src.gaussian_statistics import moments_from_canonical, energy_from_canonical

    inbox_d = node.inbox
    # n = len(inbox_d)
    # nrows = ceil(n/ncols)
    # fig = plt.figure(figsize=(3*ncols, 2*nrows))
    # gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0)
    # axs = gs.subplots(sharex=True, sharey=True)

    # First, generate a list of colours for the messages
    colors = plt.cm.rainbow(np.linspace(0, 1, len(inbox_d)))

    for i, (k, belief) in enumerate(inbox_d.items()):
        m, var = moments_from_canonical(*belief)
        local_var = var.diag_t()
        sd = local_var.sqrt()
        if trunc is None:
            trunc = len(m)
        m_trunc = m[offset:offset+trunc:step]
        sd_trunc = sd[offset:offset+trunc:step]
        x = np.arange(len(m_trunc))
        c = colors[i]
        plt.step(
            x, m_trunc,
            where='mid', color=c, alpha=0.5, label=f'{k}')
        plt.fill_between(
            x, m_trunc - sd_trunc, m_trunc + sd_trunc,
            step='mid', color=c, alpha=0.25)

    prod = node.compute_message_belief()
    m, var = moments_from_canonical(*prod)
    local_var = var.diag_t()
    sd = local_var.sqrt()
    if trunc is None:
        trunc = len(m)
    m_trunc = m[offset:offset+trunc:step]
    sd_trunc = sd[offset:offset+trunc:step]
    x = np.arange(len(m_trunc))
    plt.step(
        x, m_trunc,
        where='mid', color='black', alpha=0.25, label='prod')
    plt.fill_between(
        x, m_trunc - sd_trunc, m_trunc + sd_trunc,
        step='mid', color='black', alpha=0.5)
    if truth is not None:
        truth_trunc = truth[offset:offset+trunc:step]
        plt.step(
            x, truth_trunc,
            where='mid', color='black', alpha=0.5,
            label='truth',
            linestyle='dashed')
        error = energy_from_canonical(prod, truth, weight=False)
        energy = energy_from_canonical(prod, truth, weight=True)
        plt.title(f'{node.name} (energy {energy.item():.2f} error {error.item():.2f})')
    else:
        plt.title(node.name)
    plt.legend()
    return plt.gcf()


def node_ens_diag_plot(node, truth=None):
    """
    summarize the ensemble this this node with a diagonal sd plot

    I suspect this is no longer necessary.
    """
    ens = node.get_ens()
    m, var = moments_from_ens(ens)
    x = np.arange(len(m))
    local_var = var.diag_t()
    sd = local_var.sqrt()
    plt.step(x, m, where='mid', color='black', alpha=0.25, label='prod')
    plt.fill_between(
        x, m - sd*1.97, m + sd*1.97,
        step='mid', color='black', alpha=0.5)
    if truth is not None:
        plt.step(
            x, truth, color='black', alpha=0.5,
            label='truth',
            linestyle='dashed')
    plt.title(node.name)
    return plt.gcf()


def ens_plot(ens, ax=None, color='red', lw=0.1, alpha_scale=1.0, **kwargs):
    ens = asnumpy(ens)
    full_D = ens.shape[1]
    # the alpha exponent here chosen by trial-and-error
    alpha = min(alpha_scale * ens.shape[0] ** -0.5, 1.0)

    for line_data in ens:
        ax.plot(
            np.arange(full_D),
            line_data, color=color, alpha=alpha, lw=lw)
    # return legend handle for labeling
    return Line2D([0], [0], color=color, lw=1)


def cov_sample_plot(m, cov, n_ens=50, **kwargs):
    ens = fake_ens_from_moments(
        m, cov, n_ens=n_ens)
    ens = asnumpy(ens)
    return ens_plot(ens, **kwargs)


def cov_diag_plot(m, cov, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    m = asnumpy(m)
    cov = asnumpy(cov)
    stds = np.diagonal(cov)**0.5
    x = np.arange(len(m))
    er = ax.errorbar(x, m, yerr=stds*2, **kwargs)
    return er
