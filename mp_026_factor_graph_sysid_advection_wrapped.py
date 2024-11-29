# %%
"""
Inference in a model like this

       X0
       |
       v
Q----->X1---->X1
|      |      |
|      v      v
\----->X2---->Y2
|      |      |
|      v      v
\----->X3---->Y3
...

where the factors are high dimensional.
The target is Q, X0 is known.
"""
# %load_ext autoreload
# %autoreload 2

import time

import torch
from torch.autograd import grad
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal
from torch import optim
import torch.autograd.functional as F


import time
from pprint import pprint

import numpy as np
import submitit
import os
from dotenv import load_dotenv

from src import ensemble_bp
from src.plots import inbox_plot, cov_sample_plot, ens_plot, cov_diag_plot
from src.test_helpers import make_ball_1d, convolve_array_1d, make_blur_conv_kernel_1d, random_top_hats_basis_1d, random_cyclic_fns_1d
from src.gaussian_bp import *
from src.gaussian_statistics import moments_from_ens
from src.jobs import *

load_dotenv()

# Intermediate results we do not wish to vesion
LOG_DIR = os.getenv("LOG_DIR", "_logs")
# Outputs we wish to keep
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
# Figures we wish to keep
FIG_DIR = os.getenv("FIG_DIR", "fig")


torch.set_default_dtype(torch.float64)
import lovely_tensors as lt
lt.monkey_patch()

# in mp_026d we identify good param ranges
GENBP_BEST_MSE = {
#     'q_mse': 0.021318241953849792,
#     'q_energy': 18.85235023498535,
#     'q_loglik': 149.6551971435547,
#     'time': 18.17876124382019,
#     'memory': 77.4375,
    'method': 'genbp',
    'inf_gamma2': 0.06457715222547708,
    'inf_sigma2': 0.0693644272106558,
    'inf_eta2': 0.06670662159730129,
}
GENBP_BEST_LOGLIK =  {
#     'q_mse': 0.0288226455450058,
#     'q_energy': 67.60759735107422,
#     'q_loglik': 191.4161376953125,
#     'time': 17.509620666503906,
#     'memory': 81.9609375,
    'method': 'genbp',
    'inf_gamma2': 0.023891124805084846,
    'inf_sigma2': 0.23049552241960614,
    'inf_eta2': 0.07296746783295568,
}
GBP_BEST_MSE = {
#     'q_mse': 0.23089784383773804,
#     'q_energy': 76.16983032226562,
#     'q_loglik': float('-inf'),
#     'time': 32.514729499816895,
#     'memory': 1869.8203125,
    'method': 'gbp',
    'inf_gamma2': 0.180840414852349,
    'inf_sigma2': 0.07694863715192736,
    'inf_eta2': 0.031435215762474336,
 }
GBP_BEST_LOGLIK = {
#     'q_mse': 1.3673901557922363,
#     'q_energy': 358.09307861328125,
#     'q_loglik': -844.0155639648438,
#     'time': 21.431819677352905,
#     'memory': 1280.71484375,
    'method': 'gbp',
    'inf_gamma2': 0.037926175911161764,
    'inf_sigma2': 0.2798290057884507,
    'inf_eta2': 0.021652222458903258,
}
## NB these are not yet modified to be the best
LAPLACE_BEST_MSE = {
#     'q_mse': 0.23089784383773804,
#     'q_energy': 76.16983032226562,
#     'q_loglik': float('-inf'),
#     'time': 32.514729499816895,
#     'memory': 1869.8203125,
    'method': 'gbp',
    'inf_gamma2': 0.180840414852349,
    'inf_sigma2': 0.07694863715192736,
    'inf_eta2': 0.031435215762474336,
 }
LAPLACE_BEST_LOGLIK = {
#     'q_mse': 1.3673901557922363,
#     'q_energy': 358.09307861328125,
#     'q_loglik': -844.0155639648438,
#     'time': 21.431819677352905,
#     'memory': 1280.71484375,
    'method': 'laplace',
    'inf_gamma2': 0.037926175911161764,
    'inf_sigma2': 0.2798290057884507,
    'inf_eta2': 0.021652222458903258,
}

def run_run(
        ## process parameters
        d=128,
        circ_radius=0.125,
        conv_radius=0.125,
        downsample=2,
        obs_sigma2=0.05,  # obs noise
        tau2=0.1,  # process noise
        decay=0.7,
        shift=5,
        n_timesteps=6,
        seed=2,
        ## inference params
        method='genbp',
        callback=lambda *a: None,
        n_ens=125,
        # damping=0.25, # damping
        damping=0.0,  # no damping
        # gamma2=0.1,
        hard_damping=True,
        # inf_tau2=0.1,  # assumed process noise (so it can be >0)
        inf_gamma2=0.1, # bp noise
        q_gamma2=None,  # bp noise
        inf_sigma2=0.1,  # inference noise
        inf_eta2=0.1,  # inference noise
        max_rank=None,
        max_steps=50,
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
        job_name="",
        DEBUG_MODE=True,
        DEBUG_PLOTS=False,
        FINAL_PLOTS=False,
        SAVE_FIGURES=True,
        SHOW_FIGURES=True,
        PLOT_TITLE=False,
        return_fg=False,
        # q_ylim=(-3,7),
        q_ylim=None,
        lw=0.1,
        alpha_scale=1.0,
    ):
    print("==================")
    print(f"method {method}")
    print("==================")
    torch.manual_seed(seed)
    if q_gamma2 is None:
        q_gamma2 = inf_gamma2
    if max_rank is None:
        max_rank=n_ens
    conv = make_blur_conv_kernel_1d(int(d*conv_radius))
    # q = make_ball_1d(d, circ_radius)
    q = make_blur_conv_kernel_1d(d, scale=0.25, trunc_end=0) * d
    x0 = make_ball_1d(d, circ_radius, trunc_end=d//2)
    trunc_start = 0
    trunc_end = -1

    # simulate first step to get dimensions right
    q_prior_ens = random_cyclic_fns_1d(n_ens, d)
    # should have similar sd as height of q, or poor prior coverage
    # q_prior_ens *= q.max() / q_prior_ens.std()
    q_prior_ens *= q.std() / q_prior_ens.std() * 2

    q_len = len(q)
    x_len = len(x0)
    y_len = len(x0[...,::downsample])

    if DEBUG_PLOTS:
        from matplotlib import pyplot as plt

        plt.stairs(conv)
        plt.title('conv')
        plt.show()
        for q_p in q_prior_ens:
            plt.stairs(q_p, color='red', alpha=0.5)
        plt.stairs(q, color='red')
        plt.title('bases v truth')
        plt.show()
        for q_p in q_prior_ens:
            plt.stairs(q_p, color='red', alpha=0.5)
        for q_p in convolve_array_1d(
                q_prior_ens,
                conv,):
            plt.stairs(q_p, color='blue', alpha=0.5)
        plt.title('conv bases')
        plt.show()


    def sim_q_x__xp_noiseless(q, x):
        """
        process predictor in basic single-output form
        """
        convolved = convolve_array_1d(
            decay * x
            + (1-decay) * q,
            conv
        )
        moved = torch.roll(convolved, shift, dims=(-1,))
        return moved


    def sim_qx_xp(q, x):
        """
        process predictor
        """
        rez = sim_q_x__xp_noiseless(q, x)
        rez += torch.randn_like(rez) * tau2 ** 0.5
        return [rez]


    def q_x_xp_measurement(q_x_xp):
        q = q_x_xp[:q_len]
        x = q_x_xp[q_len:q_len+x_len]
        xp = q_x_xp[q_len+x_len:]
        return -(sim_q_x__xp_noiseless(q, x) - xp)


    def jac_q_x__xp_measurement(q_x_xp):
        q_x_xp.requires_grad_(True)
        rez = q_x_xp_measurement(q_x_xp)

        jacobian = torch.zeros(rez.numel(), q_x_xp.numel())

        for i in range(rez.numel()):
            grad_output = torch.zeros_like(rez)
            grad_output.view(-1)[i] = 1.0
            grad_input = grad(
                rez, q_x_xp, grad_outputs=grad_output, retain_graph=True)[0]

            jacobian[i, :] = grad_input.view(-1)

        return jacobian

    class QXXpModel(MeasModel):
        def __init__(self, loss: SquaredLoss, ) -> None:
            MeasModel.__init__(self, q_x_xp_measurement, jac_q_x__xp_measurement, loss)
            self.linear = False

    ## first step is special; we fix initial state

    def sim_q_x(q):
        return sim_qx_xp(q, x0)


    def q_xp_measurement(q_xp):
        q = q_xp[:q_len]
        xp = q_xp[q_len:]
        return -(sim_q_x__xp_noiseless(q, x0) - xp)


    def jac_q__xp_measurement(q_xp):
        q_xp.requires_grad_(True)
        rez = q_xp_measurement(q_xp)

        jacobian = torch.zeros(rez.numel(), q_xp.numel())

        for i in range(rez.numel()):
            grad_output = torch.zeros_like(rez)
            grad_output.view(-1)[i] = 1.0
            grad_input = grad(
                rez, q_xp, grad_outputs=grad_output, retain_graph=True)[0]

            jacobian[i, :] = grad_input.view(-1)

        return jacobian


    class QXpModel(MeasModel):
        def __init__(self, loss: SquaredLoss, ) -> None:
            MeasModel.__init__(self, q_xp_measurement, jac_q__xp_measurement, loss)
            self.linear = False


    def sim_q_qp_noiseless(q):
        """
        process predictor in basic single-output form.
        Unused
        """
        return q


    def sim_q_qp(q):
        """
        batch process predictor
        """
        return [q.clone()]


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


    def x_y_measurement(x):
        return sim_x_y_noiseless(x)

    def jac_x_y_measurement(x):
        x.requires_grad_(True)
        rez = x_y_measurement(x)

        jacobian = torch.zeros(rez.numel(), x.numel())

        for i in range(rez.numel()):
            grad_output = torch.zeros_like(rez)
            grad_output.view(-1)[i] = 1.0
            grad_input = grad(
                rez, x, grad_outputs=grad_output, retain_graph=True)[0]

            jacobian[i, :] = grad_input.view(-1)

        return jacobian

    class XYModel(MeasModel):
        def __init__(self, loss: SquaredLoss, ) -> None:
            MeasModel.__init__(self, x_y_measurement, jac_x_y_measurement, loss)
            self.linear = True


    # don't use this output! purely for dimension inference
    [_x_prior] = sim_qx_xp(q.unsqueeze(0), x0.unsqueeze(0))
    [_y_prior] = sim_x_y(_x_prior)

    del _x_prior
    del _y_prior


    def genbp_diag_plot(fg):
        from matplotlib import pyplot as plt

        plt.clf()
        plt.figure(figsize=(12,2))
        inbox_plot(
            fg.get_var_node('q'),
            truth=q
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12,2))
        inbox_plot(
            fg.get_var_node('x2'),
            truth=fg_real.get_var_node('x2').ens[0]
        )
        plt.show()
        pass

    sem = [
        (sim_q_x, ["q"], ["x1"]),  # include ancestor
        (sim_x_y, [f'x1'], [f'y1'])
    ]

    for t in range(2, n_timesteps+1):
        sem.append((sim_qx_xp, ["q", f'x{t-1}'], [f'x{t}']))
        sem.append((sim_x_y, [f'x{t}'], [f'y{t}']))

    # for generating synthetic samples
    fg_real = ensemble_bp.FactorGraph.from_sem(
        sem,
        {
            "q": q.unsqueeze(0),
        },
        sigma2=inf_sigma2,  # should not matter
        gamma2=inf_gamma2,  # should not matter
        eta2=inf_eta2,
    )
    # observations
    # fg.observe_d(dict(x0=x0))
    # fg_real.observe_d(dict(x0=x0))
    fg_real.ancestral_sample()
    gamma2s = {"q": q_gamma2 }



    if method == 'genbp':
        # genbp version
        gamma2s = {"q": q_gamma2 }
        start_time = time.time()
        if not forney_mode:
            inf_sem = sem
        else:
            inf_sem = [
                (sim_q_x, ["q"], ["x1"]),  # include ancestor
                (sim_x_y, [f'x1'], [f'y1'])
            ]
            for t in range(2, n_timesteps+1):
                if t == 2:
                    inf_sem.append((sim_q_qp, [f"q"], [f"q{t}"]))
                else:
                    inf_sem.append((sim_q_qp, [f"q{t-1}"], [f"q{t}"]))
                inf_sem.append((sim_qx_xp, [f"q{t}", f"x{t-1}"], [f"x{t}"]))
                inf_sem.append((sim_x_y, [f"x{t}"], [f"y{t}"]))
                gamma2s[f"q{t}"] = q_gamma2

        # for inference
        fg = ensemble_bp.FactorGraph.from_sem(
            inf_sem,
            {
                "q": q_prior_ens,
            },
            sigma2=inf_sigma2,
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
            belief_retain_all=belief_retain_all,
            conform_retain_all=conform_retain_all,
            conform_randomize=conform_randomize,
            conform_r_eigen_floor=conform_r_eigen_floor,
            DEBUG_MODE=DEBUG_MODE,
            verbose=10,
            atol=atol,
            rtol=rtol,
        )
        fg.ancestral_sample()
        q_node = fg.get_var_node('q')
        for t in range(1, n_timesteps+1):
            print("observing", f'y{t}')
            obs = fg_real.get_var_node(f'y{t}').get_ens().squeeze(0)
            fg.observe_d(dict(**{f'y{t}': obs}))
            x_node = fg.get_var_node(f'x{t}')

        if FINAL_PLOTS:
            from matplotlib import pyplot as plt

            legend_handles_q = []
            legend_handles_x = []
            legend_labels_q = []

            fig_q = plt.figure(1)
            fig_x = plt.figure(2)

            ax_q = fig_q.add_subplot(1, 1, 1)
            ax_x = fig_x.add_subplot(1, 1, 1)
            if PLOT_TITLE:
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

            legend_labels_q.append("prior samples")

        energies = fg.solve()

        if FINAL_PLOTS:
            legend_handles_q.append(ens_plot(q_node.get_ens(), ax=ax_q, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_handles_x.append(ens_plot(x_node.get_ens(), ax=ax_x, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_labels_q.append("posterior samples")

            q_line_handle, = ax_q.plot(q, linestyle='dashed', label="ground truth", color='black')
            legend_handles_q.append(q_line_handle)
            legend_labels_q.append("ground truth")

            ax_q.plot(q, linestyle='dashed', label="truth", color='black')
            # ax_x.plot(true_xn, linestyle='dashed', label="truth")

            if q_ylim:
                ax_q.set_ylim(*q_ylim)
            ax_q.legend(legend_handles_q, legend_labels_q)
            # ax_x.legend(legend_handles_x, legend_labels_q)
            # fig_q.show()
            # fig_x.show()
            if SAVE_FIGURES:
                os.makedirs(FIG_DIR, exist_ok=True)
                fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.pdf")
                # fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.png")
                fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.pdf")
                # fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.png")
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
        if return_fg:
            res['fg'] = fg
        return res

    elif method == 'gbp':
        ## GaBP version
        start_time = time.time()

        gbp_settings = GBPSettings(
            damping = damping,
            beta = 0.01,
            num_undamped_iters = 1,
            min_linear_iters = 1,
            # dropout = 0.0,
        )

        q_prior_mean, q_prior_cov_h = moments_from_ens(q_prior_ens, sigma2=1e-3)
        # Gaussian noise measurement model parameters:
        # q_prior_cov = torch.tensor([4.])
        # # allow variance by site
        # q_prior_cov =  q_prior_ens.var(dim=0)
        # uniform variance
        # q_prior_cov =  torch.full((q_len,), (q_prior_ens.var()))
        # full-rank variance:
        q_prior_cov = q_prior_cov_h.to_tensor()

        x_prior_mean = sim_q_x__xp_noiseless(q_prior_mean, x0)
        # x_prior_cov = torch.full((x_len,), (tau2+obs_sigma2))
        # y_prior_cov = torch.full((y_len,), obs_sigma2)

        # true_q = torch.full((q_len,), 1.0)
        true_x1 = sim_q_x__xp_noiseless(q, x0)  # <- should never be used; for display only
        true_y1 = sim_x_y_noiseless(true_x1)  # <- should never be used; for display only

        fg = FactorGraph(gbp_settings)
        q_id = fg.add_var_node(
            q_len, q_prior_mean, q_prior_cov)

        x = sim_qx_xp(q, x0)[0]  # unused, directly
        meas_y = sim_x_y(x)[0]  # data
        # meas_y1 = meas_y  # keep the first step for plotting later

        x_id = fg.add_var_node(
            x_len, x_prior_mean, torch.full((x_len,), (inf_sigma2+obs_sigma2)))

        q_x_id = fg.add_factor(
            [q_id, x_id],
            # x_prior_mean,
            torch.zeros_like(x_prior_mean),
            QXpModel(SquaredLoss(x_len, torch.full((x_len,), (inf_sigma2))))
        )
        x_y_id = fg.add_factor(
            [x_id],
            meas_y,
            XYModel(SquaredLoss(y_len, torch.full((y_len,), (inf_sigma2))))
        )

        prev_x_id = x_id

        for n in range(1, n_timesteps):
            # x = sim_qx_xp(q, x)[0]  # unused, directly
            # meas_y = sim_x_y(x)[0]  # data
            true_xn = fg_real.get_var_node(f'x{n}').ens[0]
            meas_y = fg_real.get_var_node(f'y{n}').ens[0]
            true_yn = meas_y

            x_id = fg.add_var_node(
                x_len,
                x_prior_mean,
                torch.full((x_len,), (n*inf_gamma2))  #scale cov
            )

            q_x_id = fg.add_factor(
                [q_id, prev_x_id, x_id],
                # x_prior_mean,
                torch.zeros_like(x_prior_mean),
                QXXpModel(SquaredLoss(x_len, torch.full((x_len,), (inf_sigma2))))
            )
            x_y_id = fg.add_factor(
                [x_id],
                meas_y,
                XYModel(SquaredLoss(y_len, torch.full((y_len,), (inf_sigma2+obs_sigma2))))
            )

        if FINAL_PLOTS:
            legend_handles_q = []
            legend_handles_x = []
            legend_labels_q = []

            fig_q = plt.figure(1)
            fig_x = plt.figure(2)

            ax_q = fig_q.add_subplot(1, 1, 1)
            ax_x = fig_x.add_subplot(1, 1, 1)
            if PLOT_TITLE:
                ax_q.set_title("q")
                ax_x.set_title("xn")

            q_m_est, x1_m_est, *_, xn_m_est = fg.belief_means_separately()
            q_cov_est, x1_cov_est, *_, xn_cov_est = fg.belief_covs()

            legend_handles_q.append(
                cov_sample_plot(
                    q_m_est,
                    q_cov_est,
                    n_ens=n_ens,
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
                cov_sample_plot(
                    xn_m_est,
                    xn_cov_est,
                    n_ens=n_ens,
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
            legend_labels_q.append("prior samples")

        if DEBUG_MODE:
            fg.print(brief=False)

        fg.gbp_solve(n_iters=50)
        fg.print(brief=False)
        if FINAL_PLOTS:
            q_m_est, x1_m_est, *_, xn_m_est = fg.belief_means_separately()
            q_cov_est, x1_cov_est, *_, xn_cov_est = fg.belief_covs()

            legend_handles_q.append(
                cov_sample_plot(
                    q_m_est,
                    q_cov_est,
                    n_ens=n_ens,
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
                cov_sample_plot(
                    xn_m_est,
                    xn_cov_est,
                    n_ens=n_ens,
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

            q_line_handle, = ax_q.plot(q, linestyle='dashed', label="ground truth", color='black')
            legend_handles_q.append(q_line_handle)
            legend_labels_q.append("ground truth")

            # ax_x.plot(true_xn, linestyle='dashed', label="truth")
            if q_ylim:
                ax_q.set_ylim(*q_ylim)
            ax_q.legend(legend_handles_q, legend_labels_q)
            ax_x.legend(legend_handles_x, legend_labels_q)

            # fig_q.show()
            # fig_x.show()
            if SAVE_FIGURES:
                fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.pdf")
                # fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.png")
                fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.pdf")
                # fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.png")

            if SHOW_FIGURES:
                plt.show()


        end_time = time.time()
        elapsed_time = end_time - start_time

        # fg_energy = fg.energy()
        # belief eval is fiddly; the GaBP code wants to evaluate factor-wise AFAICS
        q_m_est,  *_ = fg.belief_means_separately()
        q_cov_est, *_ = fg.belief_covs()
        q_residual = q_m_est - q
        q_mse = (q_residual**2).mean()
        q_energy = 0.5 * q_residual @ torch.inverse(q_cov_est) @ q_residual
        q_loglik = torch.tensor(float('-inf'))
        try:
            q_loglik = MultivariateNormal(q_m_est, q_cov_est).log_prob(q)
        except ValueError as e:
            # assume not PSD
            # hack: attempt to inflate diagonals by smallest eigencal so we can get a ballpark
            try:
                eigs = torch.linalg.eigvalsh(q_cov_est)
                min_eig = eigs.min()
                warnings.warn(f"q_cov_est not PSD; min eig {min_eig}")
                if min_eig < 0:
                    q_loglik = MultivariateNormal(q_m_est, q_cov_est + (-min_eig + 1e-6) * torch.eye(q_len)).log_prob(q)
                else:  #???
                    q_loglik = MultivariateNormal(q_m_est, q_cov_est + (1e-6) * torch.eye(q_len)).log_prob(q)
            except ValueError as e2:
                #doomed
                pass

        res = dict(
            # fg_mse=fg_energy.item(),
            q_mse=q_mse.item(),
            q_energy=q_energy.item(),
            q_loglik=q_loglik.item(),
            time=elapsed_time,
            n_iters=0,
        )
        if return_fg:
            res['fg'] = fg
        return res

    elif method == 'laplace':
        ## likelihood version
        start_time = time.time()
        # peak_memory_start = memory_usage(max_usage=True)

        fg = ensemble_bp.FactorGraph.from_sem(
            sem,
            {
                "q": q_prior_ens,
            },
            sigma2=inf_sigma2,
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
            belief_retain_all=belief_retain_all,
            conform_retain_all=conform_retain_all,
            conform_randomize=conform_randomize,
            conform_r_eigen_floor=conform_r_eigen_floor,
            DEBUG_MODE=DEBUG_MODE,
            verbose=10,
            atol=atol,
            rtol=rtol,
        )
        fg.ancestral_sample()
        q_node = fg.get_var_node('q')

        q_prior_mean, q_prior_cov = moments_from_ens(
            q_prior_ens, sigma2=inf_sigma2)
        # full-rank variance:
        # q_prior_cov = q_prior_cov_h.to_tensor()
        q_prior = MultivariateNormal(
            q_prior_mean, q_prior_cov.to_tensor())
        x_prior_means = []
        x_prior_covs = []
        x_priors = []
        obs = []
        y_models = []

        for t in range(1, n_timesteps+1):
            x_prior_mean, x_prior_cov = moments_from_ens(
                fg.get_var_node(f'x{t}').get_ens(),
                sigma2=inf_sigma2
            )
            x_prior_means.append(x_prior_mean)
            x_prior_covs.append(x_prior_cov)
            x_priors.append(MultivariateNormal(
                x_prior_mean, x_prior_cov.to_tensor()))

            # y_prior_mean, y_prior_cov = moments_from_ens(
            #     fg_real.get_var_node(f'y{t}'),
            #     sigma2=obs_sigma2)
            # y_prior_means.append(y_prior_mean)
            # y_prior_covs.append(y_prior_cov)
            # y_likelihood = MultivariateNormal(
            this_obs = fg_real.get_var_node(f'y{t}').get_ens().squeeze(0)
            obs.append(this_obs)
            y_model = Normal(
                loc=this_obs,
                scale=obs_sigma2**0.5
            )
            y_models.append(y_model)

        x_prior_mean = sim_q_x__xp_noiseless(q_prior_mean, x0)

        # x_prior_cov = torch.full((x_len,), (tau2+obs_sigma2))
        # y_prior_cov = torch.full((y_len,), obs_sigma2)
        def log_density(q, xs, n_timesteps):
            log_density_val = torch.zeros(1, dtype=q.dtype, device=q.device)

            # Add the log density contribution from the prior on `q`
            log_density_val += q_prior.log_prob(q).sum()
            x_prev = x0  # Assuming xs[0] is x0 and is given/known (not a variable)

            # Loop over the number of time steps
            for t in range(1, n_timesteps + 1):
                # Predict the next state from the previous state and parameter q
                x_pred = sim_q_x__xp_noiseless(q, x_prev)
                x = xs[t-1]
                # Add the log density contribution from the prior on x
                log_density_val += x_priors[t-1].log_prob(x).sum()
                # Calculate the observation from the predicted state
                y_pred = sim_x_y_noiseless(x_pred)
                # nb we could use the use-supplied state, but this will cook our chance of estimating q

                # Observation likelihood
                log_density_val += y_models[t-1].log_prob(y_pred).sum()

                # Update x_prev for the next iteration
                x_prev = x

            return log_density_val

        def compute_hessian_for_q(log_density_func, q_est, x_ests, n_timesteps):
            # Wrap the log density computation to only return the value for `q`
            def wrapped_log_density(q):
                return log_density_func(q, x_ests, n_timesteps)

            # Compute the Hessian matrix for `q`
            hessian_q = F.hessian(wrapped_log_density, q_est)
            return hessian_q

        def compute_hessian_for_x1(log_density_func, q_est, x_ests, n_timesteps):
            # Wrap the log density computation to only return the value for `q`
            def wrapped_log_density(x1):
                return log_density_func(q_est, [x1] + x_ests[1:], n_timesteps)

            # Compute the Hessian matrix for `x1`
            hessian_x1 = F.hessian(wrapped_log_density, x_ests[1])
            return hessian_x1

        def compute_hessian_for_xn(log_density_func, q_est, x_ests, n_timesteps):
            # Wrap the log density computation to only return the value for `q`
            def wrapped_log_density(xn):
                return log_density_func(q_est, x_ests[:-1]+[xn], n_timesteps)

            # Compute the Hessian matrix for `x1`
            hessian_xn = F.hessian(wrapped_log_density, x_ests[-1])
            return hessian_xn

        q_m_est = q_prior_mean.clone().detach().requires_grad_(True)
        x_m_ests = [x_prior_mean.clone().detach().requires_grad_(True) for x_prior_mean in x_prior_means]
        parameters = [q_m_est] + x_m_ests
        optimizer = optim.SGD(parameters, lr=0.01)

        if FINAL_PLOTS:
            legend_handles_q = []
            legend_handles_x = []
            legend_labels_q = []

            fig_q = plt.figure(1)
            fig_x = plt.figure(2)

            ax_q = fig_q.add_subplot(1, 1, 1)
            ax_x = fig_x.add_subplot(1, 1, 1)
            if PLOT_TITLE:
                ax_q.set_title("q")
                ax_x.set_title("xn")

            x1_m_est, *_, xn_m_est = x_m_ests

            # legend_handles_q.append(cov_sample_plot(q_m_est, q_cov_est, n_ens=n_ens, ax=ax_q, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))
            # legend_handles_x.append(cov_sample_plot(xn_m_est, xn_cov_est, n_ens=n_ens, ax=ax_x, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))
            # legend_labels_q.append("prior samples")

        # Optimization loop with early stopping
        convergence_threshold = 1e-4  # Define a threshold for early stopping
        previous_log_density = None

        for iteration in range(max_steps):
            optimizer.zero_grad()

            # Compute log density
            log_density_val = log_density(q_m_est, x_m_ests, n_timesteps)

            # Early stopping condition
            if previous_log_density is not None and abs(log_density_val.item() - previous_log_density) < convergence_threshold:
                print(f"Convergence achieved after {iteration} iterations.")
                break
            previous_log_density = log_density_val.item()

            # Perform gradient ascent
            (-log_density_val).backward()

            # Update parameters
            optimizer.step()

            # Optionally print log density to monitor progress
            print(f"Iteration {iteration}, Log Density: {log_density_val.item()}")

        hessian_q = compute_hessian_for_q(log_density, q_m_est, x_m_ests, n_timesteps)
        hessian_x1 = compute_hessian_for_x1(log_density, q_m_est, x_m_ests, n_timesteps)
        hessian_xn = compute_hessian_for_xn (log_density, q_m_est, x_m_ests, n_timesteps)
        # Compute the covariance from the inverse of the Hessian for q
        q_cov_est = torch.inverse(-hessian_q)
        x1_cov_est = torch.inverse(-hessian_x1)
        xn_cov_est = torch.inverse(-hessian_xn)
        print("Covariance matrix of q:", q_cov_est)


        if FINAL_PLOTS:
            legend_handles_q.append(
                cov_sample_plot(
                    q_m_est,
                    q_cov_est,
                    n_ens=n_ens,
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
                cov_sample_plot(
                    xn_m_est,
                    xn_cov_est,
                    n_ens=n_ens,
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

            q_line_handle, = ax_q.plot(q, linestyle='dashed', label="ground truth", color='black')
            legend_handles_q.append(q_line_handle)
            legend_labels_q.append("ground truth")

            # ax_x.plot(true_xn, linestyle='dashed', label="truth")
            if q_ylim:
                ax_q.set_ylim(*q_ylim)
            ax_q.legend(legend_handles_q, legend_labels_q)
            ax_x.legend(legend_handles_x, legend_labels_q)

            # fig_q.show()
            # fig_x.show()
            if SAVE_FIGURES:
                fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.pdf")
                # fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.png")
                fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.pdf")
                # fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.png")

            if SHOW_FIGURES:
                plt.show()


        end_time = time.time()
        # peak_memory_end = memory_usage(max_usage=True)
        elapsed_time = end_time - start_time
        # peak_memory_usage = peak_memory_end - peak_memory_start

        # fg_energy = fg.energy()
        # belief eval is fiddly; the GaBP code wants to evaluate factor-wise AFAICS
        q_m_est = q_m_est
        q_residual = q_m_est - q
        q_mse = (q_residual**2).mean()
        # q_energy = 0.5 * q_residual @ torch.inverse(q_cov_est) @ q_residual
        q_loglik = torch.tensor(float('-inf'))
        try:
            q_loglik = MultivariateNormal(q_m_est, q_cov_est).log_prob(q)
        except ValueError as e:
            try:
                eigs = torch.linalg.eigvalsh(q_cov_est)
                min_eig = eigs.min()
                max_eig = eigs.max()
                warnings.warn(f"e {e}, min eig {min_eig}, max eig {max_eig}")
                if min_eig < 0:
                    q_loglik = MultivariateNormal(q_m_est, q_cov_est + (-min_eig + 1e-6) * torch.eye(q_len)).log_prob(q)
                else:  #???
                    q_loglik = MultivariateNormal(q_m_est, q_cov_est + (1e-6) * torch.eye(q_len)).log_prob(q)
            except ValueError as e2:
                #doomed
                pass

        res = dict(
            # fg_mse=fg_energy.item(),
            q_mse=q_mse.item(),
            # q_energy=q_energy.item(),
            q_loglik=q_loglik.item(),
            time=elapsed_time,
            # memory=peak_memory_usage,
            n_iters=0,
        )
        if return_fg:
            res['fg'] = fg
        return res

    else:
        raise ValueError(f"unknown method {method}")

# %%
