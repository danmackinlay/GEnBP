# %%
"""
Inference in a model like this

W, Xhat_0 --> Xhat_1
W, Xhat_1 --> Xhat_2
...
W, Xhat_T-1 --> Xhat_T

U, X_0 --> X1
U, X_1 --> X2
...
U, X_T-1 --> X_T

X_1 --> Y_1
X_2 --> Y_2
...
X_T --> Y_T

X_1 --> Xhat_1
X_2 --> Xhat_2
...
X_T --> Xhat_T
"""
# %load_ext autoreload
# %autoreload 2

import time
from copy import deepcopy
import random

import torch
from torch.autograd import grad
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal
import torch.autograd.functional as F

from matplotlib import pyplot as plt
import cloudpickle
import gzip
import numpy as np
import os
from dotenv import load_dotenv

# from tueplots import bundles
# plt.rcParams.update(bundles.iclr2024())
from neuralop.models import TFNO, FNO

from src import ensemble_bp
from src.plots import inbox_plot, cov_sample_plot, ens_plot, cov_diag_plot
from src.ns_2d import  navier_stokes_2d_step
from src.random_fields import GaussianRF
from src.gaussian_statistics import moments_from_ens
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
from src.pretrain_no import train_model

load_dotenv()

# Intermediate results we do not wish to vesion
LOG_DIR = os.getenv("LOG_DIR", "_logs")
# Outputs we wish to keep
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
# Figures we wish to keep
FIG_DIR = os.getenv("FIG_DIR", "fig")


import lovely_tensors as lt
lt.monkey_patch()


def run_run(
        ## process parameters
        d=32,
        visc=0.01,
        delta_t=0.01,
        interval=5,
        v_noise_power=1e2,
        downsample=4,
        obs_sigma2=0.05,  # obs noise
        x_alpha=1.5,
        x_tau=1,
        q_alpha=2.5,
        q_tau=3,
        n_timesteps=6,
        seed=2,
        latent_scale=1.0,
        ## inference params
        callback=lambda *a: None,
        n_ens=30,
        damping=0.25,  # damping
        # damping=0.0,  # no damping
        # gamma2=0.1,
        hard_damping=True,
        # inf_tau2=0.1,  # assumed process noise (so it can be >0)
        inf_gamma2=0.1,  # bp noise
        q_gamma2=None,  # bp noise
        w_gamma2=None,  # bp noise
        inf_sigma2=0.5,  # inference noise
        q_sigma2=0.1,  # inference noise
        w_sigma2=0.1,  # inference noise
        inf_eta2=0.1,  # inference noise
        w_inflation=1.0,  # diffuse weight prior
        max_rank=None,
        max_steps=100,
        cvg_tol=10,
        rtol=1e-6,
        atol=1e-8,
        empty_inboxes=True,
        min_mp_steps=20,
        max_relin_int=5,
        belief_retain_all=False,
        conform_retain_all=True,
        conform_r_eigen_floor=1e-4,
        conform_randomize=True,
        ## diagnostics
        job_name="",
        DEBUG_MODE=True,
        DEBUG_PLOTS=False,
        FINAL_PLOTS=False,
        SAVE_FIGURES=False,
        SHOW_FIGURES=False,
        return_fg=False,
        # q_ylim=(-1,1),
        q_ylim=None,
        lw=0.1,
        alpha_scale=1.0,
        fno_typ="TFNO",
        fno_n_modes=16,
        fno_hidden_channels=32,
        fno_n_layers=4,
        dtype=torch.float64,
        cheat=0.0,
        sparsify_alpha=0.2,
        lateness=1.0,
        dummy=None,  #ignored
    ):
    ## Seeding is  broken for some reason?
    # seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    if max_rank is None:
        max_rank=n_ens
    # rather than start from a random model, we start from a pre-trained model, trained on a somewhat different task.
    torch.set_default_dtype(torch.float32)
    model, w_variances = train_model(
        typ=fno_typ,
        n_modes=fno_n_modes,
        hidden_channels=fno_hidden_channels,
        interval=interval,
        d=d,
        delta_t=delta_t,
        visc=visc,
        v_noise_power=v_noise_power,
        n_layers=fno_n_layers,
        alpha=x_alpha,
        tau=x_tau,
        seed=seed
    )
    normalize_model_dtypes(model, dtype)
    torch.manual_seed(seed+23)
    # deep copy original model if we want to return it
    if return_fg:
        original_model = deepcopy(model)
    torch.set_default_dtype(dtype)
    q_grf = GaussianRF(
        2, d, alpha=q_alpha, tau=q_tau)
    x_grf = GaussianRF(
        2, d, alpha=x_alpha, tau=x_tau)
    q_2d = q_grf.sample(1)[0] * latent_scale
    x0_2d = x_grf.sample(1)[0]
    # we actually need to shove a singleton channel dimension in there
    # but we make that explicit
    x0_2d = x0_2d.unsqueeze(0)

    # choose a transect to plot
    trunc_start = d
    trunc_end = d + d * 2 // 2

    q_prior_ens_2d = q_grf.sample(n_ens) * latent_scale
    # FOR TESTING ONLY initilize the prior with bias towards oracle mean
    q_prior_ens_2d += cheat * q_2d
    q, q_shape = flatten_tensor_batch(q_2d.unsqueeze(0))
    x0, x_shape = flatten_tensor_batch(x0_2d.unsqueeze(0))
    q_prior_ens, q_shape = flatten_tensor_batch(q_prior_ens_2d)

    # model = FNO(
    #     n_modes=(fno_n_modes,fno_n_modes),
    #     hidden_channels=fno_hidden_channels,
    #     in_channels=1,
    #     out_channels=1,
    #     dtype=dtype,
    # )
    # model = TFNO(
    #     n_modes=(fno_n_modes,fno_n_modes), hidden_channels=fno_hidden_channels,
    #     in_channels=1,
    #     out_channels=1,
    #     factorization='tucker',
    #     implementation='factorized',
    #     rank=0.05)


    w_vector, w_shapes = flatten_model_parameters(model)
    w = w_vector.unsqueeze(0)
    # generate a batch of size `(n_ens, len(w))` perturbed by a noise comparable to the variance of the data itself.
    w_var = flatten_gradient_variances(w_variances, w_shapes)
    sparsify_alpha = torch.tensor(sparsify_alpha)
    beta = sparsify_alpha / w_var
    gamma_dist = torch.distributions.Gamma(
        concentration=sparsify_alpha, rate=beta)
    w_var_sparse = gamma_dist.sample((n_ens,))
    w_sd = w_var_sparse.sqrt() * w_inflation
    ## null out the first `(1-lateness)*len(w_var)` weights
    w_sd[:, :int((1-lateness)*len(w_var))] = 0.0
    print(w_sd)

    w_prior_ens = w + torch.randn(n_ens, w_vector.numel()) * w_sd

    def sim_qx_xp(q, x):
        """
        process predictor
        """
        x_t = unflatten_tensor_batch(x, x_shape)
        q_t = unflatten_tensor_batch(q, q_shape)
        res = navier_stokes_2d_step(
            x_t,
            f=q_t.unsqueeze(1),  # add channel dimension
            visc=visc,
            delta_t=delta_t,
            interval=interval,
            v_noise_power=v_noise_power,
        )
        res, res_shape = flatten_tensor_batch(res)
        return [res]

    def sim_q_x(q):
        """
        first step is special; we fix initial state
        # TODO: generalize to unknown initial state
        # """
        return sim_qx_xp(q, x0)

    def sim_x_y_noiseless(x):
        """
        Obs model
        """
        x = x[...,::downsample]  # downsample
        return x

    def sim_x_y(x):
        """
        Obs model
        """
        rez = sim_x_y_noiseless(x)
        rez += torch.randn_like(rez) * obs_sigma2** 0.5
        return [rez]

    def copy(x):
        """
        enforce xhat = x
        """
        return [x.clone()]

    def sim_wxhat_xhatp(w, xhat):
        """
        emulated process predictor
        """
        xp_vec_batch, output_shape = batched_vector_model(
            xhat, w, x_shape, w_shapes, model
        )
        return [xp_vec_batch]

    def sim_wxhat(w):
        """
        first step is special; we fix initial state
        TODO: generalize to unknown initial state
        """

        xp_vec_batch, output_shape = batched_vector_model(
            x0, w, x_shape, w_shapes, model
        )
        return [xp_vec_batch]

    sem = [
        (sim_q_x, ["q"], ["x1"]),  # include ancestor
        (sim_x_y, [f'x1'], [f'y1']),
        (None, [f'x1'], [f'xhat1']),
        (sim_wxhat, [f'w'], [f'xhat1']),
    ]

    for t in range(2, n_timesteps+1):
        sem.append((sim_qx_xp, ["q", f'x{t-1}'], [f'x{t}']))
        sem.append((sim_x_y, [f'x{t}'], [f'y{t}']))
        sem.append((None, [f'x{t}'], [f'xhat{t}']))
        sem.append((sim_wxhat_xhatp, [f'w', f'xhat{t-1}'], [f'xhat{t}']))

    # for generating synthetic samples
    fg_real = ensemble_bp.FactorGraph.from_sem(
        sem,
        {"q": q, "w": w},
        sigma2=inf_sigma2,  # should not matter
        gamma2=inf_gamma2,  # should not matter
        eta2=inf_eta2,
    )
    # observations
    # fg.observe_d(dict(x0=x0))
    # fg_real.observe_d(dict(x0=x0))
    fg_real.ancestral_sample()
    sigma2s = {
        "q": q_sigma2 if q_sigma2 is not None else inf_sigma2,
        "w": w_sigma2 if w_sigma2 is not None else inf_sigma2,
    }
    gamma2s = {
        "q": q_gamma2 if q_gamma2 is not None else inf_gamma2,
        "w": w_gamma2 if w_gamma2 is not None else inf_gamma2,
    }
    start_time = time.time()
    ## Generate Forney-style BP graph
    inf_sem = [
        (sim_q_x, ["q"], ["x1"]),  # include ancestor
        (sim_x_y, [f"x1"], [f"y1"]),
        (None, [f"x1"], [f"xhat1"]),
        (sim_wxhat, [f"w"], [f"xhat1"]),
    ]
    for t in range(2, n_timesteps+1):
        gamma2s[f"q{t}"] = q_gamma2 if q_gamma2 is not None else inf_gamma2
        gamma2s[f"w{t}"] = w_gamma2 if w_gamma2 is not None else inf_gamma2
        sigma2s[f"q{t}"] = q_sigma2 if q_sigma2 is not None else inf_sigma2
        sigma2s[f"w{t}"] = w_sigma2 if w_sigma2 is not None else inf_sigma2
        if t == 2:
            inf_sem.append((copy, [f"q"], [f"q{t}"]))
            inf_sem.append((copy, [f"w"], [f"w{t}"]))
        else:
            inf_sem.append((copy, [f"q{t-1}"], [f"q{t}"]))
            inf_sem.append((copy, [f"w{t-1}"], [f"w{t}"]))
        inf_sem.append((sim_qx_xp, [f"q{t}", f"x{t-1}"], [f"x{t}"]))
        inf_sem.append((sim_x_y, [f"x{t}"], [f"y{t}"]))
        inf_sem.append((None, [f"x{t}"], [f"xhat{t}"]))
        inf_sem.append((sim_wxhat_xhatp,
            [f"w{t}", f"xhat{t-1}"], [f"xhat{t}"]))

    def genbp_diag_plot(fg):
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("q"),
            truth=q.squeeze(0),
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("q2"),
            truth=q.squeeze(0),
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("x3"),
            truth=fg_real.get_var_node("x3").ens[0],
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("x4"),
            truth=fg_real.get_var_node("x4").ens[0],
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("xhat3"),
            truth=fg_real.get_var_node("x3").ens[0],
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("xhat4"),
            truth=fg_real.get_var_node("x4").ens[0],
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12, 2))
        inbox_plot(
            fg.get_var_node("w2"),
            truth=fg_real.get_var_node("w").ens[0],
            trunc_start=trunc_start,
            trunc_end=trunc_end,
            step=1,
        )
        plt.show()

    # for inference
    fg = ensemble_bp.FactorGraph.from_sem(
        inf_sem,
        {
            "q": q_prior_ens,
            "w": w_prior_ens,
        },
        sigma2=inf_sigma2,
        sigma2s=sigma2s,
        gamma2=inf_gamma2,
        gamma2s=gamma2s,
        eta2=inf_eta2,
        max_rank=max_rank,
        damping=damping,
        max_steps=max_steps,
        hard_damping=hard_damping,
        callback=genbp_diag_plot if DEBUG_PLOTS else callback,
        empty_inboxes=empty_inboxes,
        min_mp_steps=min_mp_steps,
        max_relin_int=max_relin_int,
        belief_retain_all=belief_retain_all,
        conform_retain_all=conform_retain_all,
        conform_randomize=conform_randomize,
        conform_r_eigen_floor=conform_r_eigen_floor,
        DEBUG_MODE=DEBUG_MODE,
        verbose=10,
        atol=atol,
        rtol=rtol,
        cvg_tol=cvg_tol,
    )
    for t in range(1, n_timesteps+1):
        # Don't update simulator from emulator
        fg.get_factor_node(f'__x{t}__xhat{t}').blocked = (f'x{t}',)


    fg.ancestral_sample()
    q_node = fg.get_var_node('q')

    for t in range(1, n_timesteps+1):
        print("observing", f'y{t}')
        obs = fg_real.get_var_node(f'y{t}').get_ens().squeeze(0)
        fg.observe_d(dict(**{f'y{t}': obs}))
        x_node = fg.get_var_node(f'x{t}')

    if FINAL_PLOTS:
        legend_handles_q = []
        legend_handles_x = []
        legend_labels_q = []

        fig_q = plt.figure(1)
        fig_x = plt.figure(2)

        ax_q = fig_q.add_subplot(1, 1, 1)
        ax_x = fig_x.add_subplot(1, 1, 1)

        ax_q.set_title("q")
        ax_x.set_title("xn")

        legend_handles_q.append(
            ens_plot(
                q_node.get_ens(),
                trunc_start=trunc_start,
                trunc_end=trunc_end,
                ax=ax_q,
                color="red",
                ecolor="red",
                label="prior samples",
                lw=lw,
                alpha_scale=alpha_scale,
            )
        )
        legend_handles_x.append(
            ens_plot(
                x_node.get_ens(),
                trunc_start=trunc_start,
                trunc_end=trunc_end,
                ax=ax_x,
                color="red",
                ecolor="red",
                label="prior samples",
                lw=lw,
                alpha_scale=alpha_scale,
            )
        )

        legend_labels_q.append("prior")

    energies = fg.solve()

    if FINAL_PLOTS:
        legend_handles_q.append(
            ens_plot(
                q_node.get_ens(),
                trunc_start=trunc_start,
                trunc_end=trunc_end,
                ax=ax_q,
                color="blue",
                ecolor="blue",
                label="posterior samples",
                lw=lw,
                alpha_scale=alpha_scale,
            )
        )
        legend_handles_x.append(
            ens_plot(
                x_node.get_ens(),
                trunc_start=trunc_start,
                trunc_end=trunc_end,
                ax=ax_x,
                color="blue",
                ecolor="blue",
                label="posterior samples",
                lw=lw,
                alpha_scale=alpha_scale,
            )
        )
        legend_labels_q.append("posterior samples")

        q_line_handle, = ax_q.plot(
            q[0, trunc_start:trunc_end],
            linestyle='dashed',
            label="ground truth", color='black')
        legend_handles_q.append(q_line_handle)
        legend_labels_q.append("ground truth")

        true_xn = fg_real.get_var_node(f'x{n_timesteps}').get_ens().squeeze(0)
        ax_x.plot(true_xn[trunc_start:trunc_end], linestyle="dashed", label="truth")

        if q_ylim is not None:
            ax_q.set_ylim(*q_ylim)
        ax_q.legend(legend_handles_q, legend_labels_q)
        # ax_x.legend(legend_handles_x, legend_labels_q)
        # fig_q.show()
        # fig_x.show()
        if SAVE_FIGURES:
            os.makedirs(FIG_DIR, exist_ok=True)
            base_name = f"{job_name}_{seed}_visc{visc:.4f}_d{d}_nens{n_ens}"
            fig_q.savefig(f"{FIG_DIR}/{base_name}_jq_update.pdf")
            # fig_q.savefig(f"{FIG_DIR}/{base_name}_jq_update.png")
            fig_x.savefig(f"{FIG_DIR}/{base_name}_xn_update.pdf")
            # fig_x.savefig(f"{FIG_DIR}/{base_name}_xn_update.png")

        if SHOW_FIGURES:
            plt.show()

    print(energies)
    end_time = time.time()

    elapsed_time = end_time - start_time
    # q_m_est, q_cov_est = q_node.get_moments_belief()
    q_residual = q_node.get_residual(q)
    q_mse = q_node.get_mse(q)
    q_energy =  q_node.get_ens_energy(q)
    q_loglik = q_node.get_loglik(q)

    res = dict(
        # fg_mse=fg_energy.item(),
        q_mse=q_mse.item(),
        q_energy=q_energy.item(),
        q_loglik=q_loglik.item(),
        time=elapsed_time,
        n_iters=0,
    )
    if return_fg and job_name:
        # create a dict with all the thing we want to keep and save them efficiently.
        outpath = f"{OUTPUT_DIR}/_{job_name}_res.pkl.gz"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        extras = {
            "fg": fg,
            "fg_real": fg_real,
            "sim_wxhat_xhatp": sim_wxhat_xhatp,
            "sim_qx_xp": sim_qx_xp,
            "q_2d": q_2d,
            "x0_2d": x0_2d,
            "q_grf": q_grf,
            "x_grf": x_grf,
            "model": model,
            "original_model": original_model,
            "w_prior_ens": w_prior_ens,
        }
        res["outpath"] = outpath
        try:
            with gzip.open(outpath, "wb") as f:
                cloudpickle.dump(extras, f)
        except Exception as e:
            print(f"Failed to save extras: {e}")
        return res, extras
    return res

# %%
