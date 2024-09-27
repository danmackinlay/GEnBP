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
Q parameterises a neural network that generates X1 from X0.
"""
# %load_ext autoreload
# %autoreload 2

import time
from memory_profiler import memory_usage

import torch
from torch.autograd import grad
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal

from matplotlib import pyplot as plt
import time
from memory_profiler import memory_usage
from pprint import pprint

import numpy as np
import submitit
import os
from dotenv import load_dotenv

from neuralop.models import FNO, TFNO

# from tueplots import bundles
# plt.rcParams.update(bundles.icmlr2025())

from src import ensemble_bp
from src.plots import inbox_plot, cov_sample_plot, ens_plot, cov_diag_plot
from src.gaussian_bp import *
from src.ns_2d import convert_1d_to_2d, convert_2d_to_1d, navier_stokes_2d_step_vector_form, navier_stokes_2d_step
from src.random_fields import GaussianRF
from src.math_helpers import convert_1d_to_2d, convert_2d_to_1d
from src.gaussian_statistics import moments_from_ens

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


def run_run(
        ## process parameters
        d=128,
        visc = 0.5,
        delta_t = 0.01,
        interval = 2,
        v_noise_power = 1e2,
        downsample=2,
        obs_sigma2=0.05,  # obs noise
        x_alpha=1.5,
        x_tau=1,
        q_alpha=2.5,
        q_tau=3,
        n_timesteps=6,
        seed=2,
        ## inference params
        method='genbp',
        callback=lambda *a: None,
        n_ens=125,
        damping=0.25, # damping
        # damping=0.0,  # no damping
        # gamma2=0.1,
        hard_damping=True,
        # inf_tau2=0.1,  # assumed process noise (so it can be >0)
        inf_gamma2=0.1, # bp noise
        q_gamma2=None,  # bp noise
        inf_sigma2=0.1,  # inference noise
        inf_eta2=0.1,  # inference noise
        max_rank=None,
        max_steps=50,
        cvg_tol=-1,
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
        SAVE_FIGURES=False,
        return_fg=False,
        q_ylim=(-1,1),
        lw=0.1,
        alpha_scale=1.0,
        max_floats = 1024*1024*1024, # 8gb panic limit
    ):
    torch.manual_seed(seed)
    if q_gamma2 is None:
        q_gamma2 = inf_gamma2
    if max_rank is None:
        max_rank=n_ens

    biggest_array = max(d**2 * n_ens*3, d**4) * n_timesteps
    if biggest_array > max_floats:
        raise ValueError(f"memory limit exceeded: {biggest_array} > {max_floats}")
    q_grf = GaussianRF(
        2, d, alpha=q_alpha, tau=q_tau)
    x_grf = GaussianRF(
        2, d, alpha=x_alpha, tau=x_tau)
    q_2d = q_grf.sample(n_ens)[0]
    x0_2d = x_grf.sample(1)[0]

    q_prior_ens_2d = q_grf.sample(n_ens)
    q = convert_2d_to_1d(q_2d)
    x0 = convert_2d_to_1d(x0_2d)
    q_prior_ens = convert_2d_to_1d(q_prior_ens_2d)

    q_len = len(q)
    x_len = len(x0)
    y_len = len(x0[...,::downsample])

    if DEBUG_PLOTS:
        for q_p in q_prior_ens:
            plt.stairs(q_p, color='red', alpha=0.5)
        plt.stairs(q, color='red')
        plt.title('bases v truth')
        plt.show()

    def sim_q_x__xp_noiseless(q, x):
        """
        process predictor in basic single-output form
        """
        return navier_stokes_2d_step_vector_form(
            x, q,
            visc=visc, delta_t=delta_t, interval=interval, v_noise_power=0.0)


    def sim_qx_xp(q, x):
        """
        process predictor
        """
        return [
            navier_stokes_2d_step_vector_form(
                x, q,
                visc=visc, delta_t=delta_t, interval=interval,
                v_noise_power=v_noise_power)
        ]

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

    def genbp_diag_plot(fg):
        plt.clf()
        plt.figure(figsize=(12,2))
        fig = inbox_plot(
            fg.get_var_node('q'),
            truth=q,
            trunc=d,
            offset=d*2//2,
            step=1,
        )
        plt.show()
        plt.clf()
        plt.figure(figsize=(12,2))
        inbox_plot(
            fg.get_var_node('x2'),
            truth=fg_real.get_var_node('x2').ens[0],
            trunc=d,
            offset=d*2//2,
            step=1,
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

    # if DEBUG_MODE:
    #     ## Diagnostic plots; set downsample=1
    #     for name, node in fg_real.var_nodes.items():
    #         plt.clf()
    #         this_truth = node.get_ens().squeeze(0)
    #         plt.figure()
    #         this_truth_2d = convert_1d_to_2d(this_truth)
    #         plt.imshow(this_truth_2d)
    #         #colorbar
    #         plt.colorbar()
    #         plt.title(name)
    #         plt.show()


    if method == 'genbp':
        # genbp version
        gamma2s = {"q": q_gamma2 }
        start_time = time.time()
        peak_memory_start = memory_usage(max_usage=True)
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
            cvg_tol=cvg_tol,
        )
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

            legend_handles_q.append(ens_plot(q_node.get_ens(), ax=ax_q, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))
            legend_handles_x.append(ens_plot(x_node.get_ens(), ax=ax_x, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))

            legend_labels_q.append("prior")

        energies = fg.solve()

        if FINAL_PLOTS:
            legend_handles_q.append(ens_plot(q_node.get_ens(), ax=ax_q, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_handles_x.append(ens_plot(x_node.get_ens(), ax=ax_x, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_labels_q.append("posterior samples")

            q_line_handle, = ax_q.plot(q, linestyle='dashed', label="ground truth", color='black')
            legend_handles_q.append(q_line_handle)
            legend_labels_q.append("ground truth")

            ax_q.plot(q, linestyle='dashed', label="truth", color='black')
            true_xn = fg_real.get_var_node(f'x{n_timesteps}').get_ens().squeeze(0)
            ax_x.plot(true_xn, linestyle='dashed', label="truth")

            ax_q.set_ylim(*q_ylim)
            ax_q.legend(legend_handles_q, legend_labels_q)
            # ax_x.legend(legend_handles_x, legend_labels_q)
            # fig_q.show()
            # fig_x.show()
            if SAVE_FIGURES:
                os.makedirs(FIG_DIR, exist_ok=True)
                fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.pdf")
                fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.pdf")

            plt.show()

        print(energies)
        end_time = time.time()
        peak_memory_end = memory_usage(max_usage=True)

        elapsed_time = end_time - start_time
        peak_memory_usage = peak_memory_end - peak_memory_start
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
            memory=peak_memory_usage,
            n_iters=0,
        )
        if return_fg:
            res['fg'] = fg
        return res

    elif method == 'gbp':
        ## GaBP version
        start_time = time.time()
        peak_memory_start = memory_usage(max_usage=True)

        gbp_settings = GBPSettings(
            damping = damping,
            beta = 0.01,
            num_undamped_iters = 1,
            min_linear_iters = 1,
            # dropout = 0.0,
        )

        q_prior_mean, q_prior_cov_h = moments_from_ens(q_prior_ens, sigma2=inf_sigma2)

        # Gaussian noise measurement model parameters:
        # q_prior_cov = torch.tensor([4.])
        # # allow variance by site
        # q_prior_cov =  q_prior_ens.var(dim=0)
        # uniform variance
        # q_prior_cov =  torch.full((q_len,), (q_prior_ens.var()))
        # full rank prior:
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

            ax_q.set_title("q")
            ax_x.set_title("xn")

            q_m_est, x1_m_est, *_, xn_m_est = fg.belief_means_separately()
            q_cov_est, x1_cov_est, *_, xn_cov_est = fg.belief_covs()

            legend_handles_q.append(cov_sample_plot(q_m_est, q_cov_est, n_ens=n_ens, ax=ax_q, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))
            legend_handles_x.append(cov_sample_plot(xn_m_est, xn_cov_est, n_ens=n_ens, ax=ax_x, color='red', ecolor='red', label="prior samples", lw=lw, alpha_scale=alpha_scale))
            legend_labels_q.append("prior samples")

        if DEBUG_MODE:
            fg.print(brief=False)

        fg.gbp_solve(n_iters=50)
        fg.print(brief=False)
        if FINAL_PLOTS:
            q_m_est, x1_m_est, *_, xn_m_est = fg.belief_means_separately()
            q_cov_est, x1_cov_est, *_, xn_cov_est = fg.belief_covs()

            legend_handles_q.append(cov_sample_plot(q_m_est, q_cov_est, n_ens=n_ens, ax=ax_q, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_handles_x.append(cov_sample_plot(xn_m_est, xn_cov_est, n_ens=n_ens, ax=ax_x, color='blue', ecolor='blue', label="posterior samples", lw=lw, alpha_scale=alpha_scale))
            legend_labels_q.append("posterior samples")

            q_line_handle, = ax_q.plot(q, linestyle='dashed', label="ground truth", color='black')
            legend_handles_q.append(q_line_handle)
            legend_labels_q.append("ground truth")

            # ax_x.plot(true_xn, linestyle='dashed', label="truth")

            ax_q.set_ylim(*q_ylim)
            ax_q.legend(legend_handles_q, legend_labels_q)
            ax_x.legend(legend_handles_x, legend_labels_q)

            # fig_q.show()
            # fig_x.show()
            if SAVE_FIGURES:
                fig_q.savefig(f"{FIG_DIR}/{job_name}_{seed}_jq_update.pdf")
                fig_x.savefig(f"{FIG_DIR}/{job_name}_{seed}_xn_update.pdf")

            plt.show()

        end_time = time.time()
        peak_memory_end = memory_usage(max_usage=True)
        elapsed_time = end_time - start_time
        peak_memory_usage = peak_memory_end - peak_memory_start

        # fg_energy = fg.energy()
        # belief eval is fiddly; the GaBP code wants to evaluate factor-wise AFAICS
        q_m_est,  *_ = fg.belief_means_separately()
        q_cov_est, *_ = fg.belief_covs()
        q_residual = q_m_est - q
        q_mse = (q_residual**2).mean()
        q_energy = 0.5 * q_residual @ torch.inverse(q_cov_est) @ q_residual
        try:
            q_loglik = MultivariateNormal(q_m_est, q_cov_est).log_prob(q)
        except ValueError as e:
            # assume not PSD
            q_loglik = torch.tensor(float('-inf'))


        res = dict(
            # fg_mse=fg_energy.item(),
            q_mse=q_mse.item(),
            q_energy=q_energy.item(),
            q_loglik=q_loglik.item(),
            time=elapsed_time,
            memory=peak_memory_usage,
            n_iters=0,
        )
        if return_fg:
            res['fg'] = fg
        return res

    else:
        raise ValueError(f"unknown method {method}")

# %%
