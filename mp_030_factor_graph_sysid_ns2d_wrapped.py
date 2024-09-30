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

# from tueplots import bundles
# plt.rcParams.update(bundles.iclr2024())

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
        # langevin`` parameters
        langevin_step_size=0.001,
        langevin_num_samples=5000,
        langevin_burn_in=1000,
        langevin_thinning=10,
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

    elif method == 'laplace':
        ## likelihood version
        gamma2s = {"q": q_gamma2 }
        start_time = time.time()
        peak_memory_start = memory_usage(max_usage=True)

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
            cvg_tol=cvg_tol,
        )
        fg.ancestral_sample()
        q_node = fg.get_var_node('q')

        q_prior_mean, q_prior_cov = moments_from_ens(
            q_prior_ens, sigma2=inf_sigma2)
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

            this_obs = fg_real.get_var_node(f'y{t}').get_ens().squeeze(0)
            obs.append(this_obs)
            y_model = Normal(
                loc=this_obs,
                scale=obs_sigma2**0.5
            )
            y_models.append(y_model)

        x_prior_mean = sim_q_x__xp_noiseless(q_prior_mean, x0)

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
            # Wrap the log density computation to only return the value for `x1`
            def wrapped_log_density(x1):
                return log_density_func(q_est, [x1] + x_ests[1:], n_timesteps)

            # Compute the Hessian matrix for `x1`
            hessian_x1 = F.hessian(wrapped_log_density, x_ests[0])
            return hessian_x1

        def compute_hessian_for_xn(log_density_func, q_est, x_ests, n_timesteps):
            # Wrap the log density computation to only return the value for `xn`
            def wrapped_log_density(xn):
                return log_density_func(q_est, x_ests[:-1]+[xn], n_timesteps)

            # Compute the Hessian matrix for `xn`
            hessian_xn = F.hessian(wrapped_log_density, x_ests[-1])
            return hessian_xn

        q_m_est = q_prior_mean.clone().detach().requires_grad_(True)
        x_m_ests = [x_prior_mean.clone().detach().requires_grad_(True) for x_prior_mean in x_prior_means]
        parameters = [q_m_est] + x_m_ests
        optimizer = torch.optim.SGD(parameters, lr=0.01)

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

            x1_m_est = x_m_ests[0]
            xn_m_est = x_m_ests[-1]

            # We can include prior plots if desired.

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
            # Plot posterior samples
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

        # Compute residuals and MSE
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
            memory=peak_memory_usage,
            n_iters=iteration+1,
        )
        if return_fg:
            res['fg'] = fg
        return res
    elif method == 'langevin':
        # Set up the Langevin sampler parameters
        step_size = langevin_step_size
        num_samples = langevin_num_samples
        burn_in = langevin_burn_in
        thinning = langevin_thinning
        # num_samples = 5000
        # burn_in = 1000
        # thinning = 10
        # step_size = 0.001  # Adjust step size as needed

        ## Langevin sampler version
        gamma2s = {"q": q_gamma2}
        start_time = time.time()
        peak_memory_start = memory_usage(max_usage=True)
        # for inference
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
            cvg_tol=cvg_tol,
        )
        fg.ancestral_sample()
        q_node = fg.get_var_node('q')

        # Initialize priors and observations (reuse from 'laplace' method)
        q_prior_mean, q_prior_cov = moments_from_ens(
            q_prior_ens, sigma2=inf_sigma2)
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

            this_obs = fg_real.get_var_node(f'y{t}').get_ens().squeeze(0)
            obs.append(this_obs)
            y_model = Normal(
                loc=this_obs,
                scale=obs_sigma2**0.5
            )
            y_models.append(y_model)

        # Define the log_density function (same as in 'laplace')
        def log_density(q, xs, n_timesteps):
            log_density_val = torch.zeros(1, dtype=q.dtype, device=q.device)

            # Add the log density contribution from the prior on `q`
            log_density_val += q_prior.log_prob(q).sum()
            x_prev = x0  # x0 is known and fixed

            # Loop over the number of time steps
            for t in range(1, n_timesteps + 1):
                # Predict the next state from the previous state and parameter q
                x_pred = sim_q_x__xp_noiseless(q, x_prev)
                x = xs[t-1]
                # Add the log density contribution from the prior on x
                log_density_val += x_priors[t-1].log_prob(x).sum()
                # Calculate the observation from the predicted state
                y_pred = sim_x_y_noiseless(x_pred)

                # Observation likelihood
                log_density_val += y_models[t-1].log_prob(y_pred).sum()

                # Update x_prev for the next iteration
                x_prev = x

            return log_density_val

        # Initialize q and x_states
        q_current = q_prior_mean.clone().detach().requires_grad_(True)
        x_states_current = [x_prior_mean.clone().detach().requires_grad_(True) for x_prior_mean in x_prior_means]

        # Store samples
        samples_q = []
        samples_x_states = []

        total_iterations = burn_in + num_samples * thinning

        for iteration in range(total_iterations):
            # Zero gradients
            for x in x_states_current:
                if x.grad is not None:
                    x.grad.zero_()
            if q_current.grad is not None:
                q_current.grad.zero_()

            # Compute log density
            log_density_val = log_density(q_current, x_states_current, n_timesteps)

            # Compute gradients
            log_density_val.backward()

            # Perform Langevin update
            with torch.no_grad():
                # Update q
                grad_q = q_current.grad
                noise_q = torch.randn_like(q_current)
                q_new = q_current + (step_size / 2) * grad_q + torch.sqrt(torch.tensor(step_size)) * noise_q

                # Update x_states
                x_states_new = []
                for x, grad_x in zip(x_states_current, [x.grad for x in x_states_current]):
                    noise_x = torch.randn_like(x)
                    x_new = x + (step_size / 2) * grad_x + torch.sqrt(torch.tensor(step_size)) * noise_x
                    x_states_new.append(x_new)

                # Detach and set requires_grad
                q_current = q_new.detach().clone().requires_grad_(True)
                x_states_current = [x_new.detach().clone().requires_grad_(True) for x_new in x_states_new]

            # Collect samples after burn-in and according to thinning interval
            if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
                samples_q.append(q_current.detach().clone())
                samples_x_states.append([x.detach().clone() for x in x_states_current])

            # Optionally print progress
            if (iteration + 1) % 1000 == 0:
                print(f"Iteration {iteration +1}/{total_iterations}")

        # Stack samples
        samples_q = torch.stack(samples_q)
        # samples_x_states is a list of lists, need to stack properly
        samples_x_states = [torch.stack([x_states[i] for x_states in samples_x_states]) for i in range(n_timesteps)]

        # Compute elapsed time and memory usage
        end_time = time.time()
        peak_memory_end = memory_usage(max_usage=True)
        elapsed_time = end_time - start_time
        peak_memory_usage = peak_memory_end - peak_memory_start

        # Compute statistics for q
        samples_q_tensor = samples_q
        q_m_est = samples_q_tensor.mean(dim=0)
        q_cov_est = torch.cov(samples_q_tensor.T)

        # Compute q_mse and q_loglik
        q_residual = q_m_est - q
        q_mse = (q_residual**2).mean()
        try:
            q_loglik = MultivariateNormal(q_m_est, q_cov_est).log_prob(q)
        except ValueError as e:
            # Handle covariance not PSD
            q_loglik = torch.tensor(float('-inf'))

        res = dict(
            q_mse=q_mse.item(),
            q_loglik=q_loglik.item(),
            time=elapsed_time,
            memory=peak_memory_usage,
            n_iters=total_iterations,
        )
        if return_fg:
            res['fg'] = fg
        return res
    else:
        raise ValueError(f"unknown method {method}")

# %%
