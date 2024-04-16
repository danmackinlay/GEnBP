import warnings
import torch
from torch.distributions import LowRankMultivariateNormal

from ..gaussian_statistics import mean_dev_from_ens, \
    moments_from_ens, \
    moments_from_canonical, \
    canonical_from_ens, \
    fake_ens_from_canonical, \
    conform_ensemble
from ..utils import isscalar


def variable_product_message(messages):
    """
    Concatenate a list of var belief messages into a single message.
    This is its own function because it is simple.

    A more efficient implementation might economise on memory allocation
    """
    es = [e for e, _ in messages]
    precs = [prec for _, prec in messages]
    e = torch.sum(torch.stack(es), dim=0)
    # TODO: inefficient
    prec = precs.pop(0)
    for p in precs:
        prec = prec + p
    return e, prec


class VarNode:
    """
    A variable node in the factor graph.
    """
    def __init__(
            self, name, factor_nodes={}, ens=None,
            sigma2=1.0, gamma2=None, eta2=None, fg=None):
        self.name = name  # should var names be on the node?
        self.inbox = {}
        self.ens = ens  # just a matrix
        self.observation = None
        self.factor_nodes = {}
        self.belief = None
        self.moments_belief = None
        self.sigma2 = torch.as_tensor(sigma2)
        if gamma2 is None:
            gamma2 = sigma2
        self.gamma2 = torch.as_tensor(gamma2)
        if eta2 is None:
            eta2 = gamma2
        self.eta2 = torch.as_tensor(eta2)
        self.fg = fg
        for k, v in factor_nodes.items():
            self.add_factor_node(k, v)

    def set_fg(self, fg):
        self.fg = fg

    def is_debug(self):
        return self.fg.get_setting('DEBUG_MODE', False)

    def diagnosis(self):
        """
        Return a dict of diagnostic information about the var node.
        """
        inbox_beliefs = {}
        for k, bel in self.inbox.items():
            e, prec = bel
            inbox_beliefs[k] = dict(
                e_shape=tuple(e.shape),
                prec=prec.diagnosis(),
            )

        diagnosis = dict(
            ens_shape=tuple(self.ens.shape),
            sigma2=self.sigma2,
            gamma2=self.gamma2,
        )
        if self.belief is not None:
            diagnosis['prec'] = self.belief[1].diagnosis()
        if self.is_observed():
            diagnosis['observation_shape'] = self.observation.shape
        for k, v in self.inbox.items():
            diagnosis['inbox'] = inbox_beliefs
        return diagnosis

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

    def device(self):
        return self.ens.device

    def scale_sigma2(self, scale):
        self.sigma2 *= scale

    def scale_gamma2(self, scale):
        self.gamma2 *= scale

    def get_sigma2(self, realized=False):
        """
        return the sigma2 of the variable
        if `realized`, we make sure it is a vector, upcasting if necessary.
        """
        sigma2 = self.sigma2
        if isscalar(sigma2):
            D = self.get_dim()

        if not realized:
            return sigma2
        sigma2 = torch.as_tensor(sigma2)
        if sigma2.dim() == 0:  # sigma2 is a scalar tensor
            sigma2 = sigma2.expand(D)
        elif sigma2.dim() == 1 and sigma2.shape[0] == D:  # sigma2 is a vector of length D
            pass  # do nothing
        else:
            raise ValueError("Input sigma2 is not compatible")
        return sigma2

    def get_gamma2(self, realized=False):
        """
        return the gamma2 of the variable
        if `realized`, we make sure it is a vector, upcasting if necessary.
        """
        gamma2 = self.gamma2
        if isscalar(gamma2):
            D = self.get_dim()

        if not realized:
            return gamma2
        gamma2 = torch.as_tensor(gamma2)
        if gamma2.dim() == 0:  # gamma2 is a scalar tensor
            gamma2 = gamma2.expand(D)
        elif gamma2.dim() == 1 and gamma2.shape[0] == D:
            # gamma2 is a vector of length D
            pass  # do nothing
        else:
            raise ValueError("Input gamma2 is not compatible")
        return gamma2

    def get_eta2(self, realized=False):
        """
        return the eta2 of the variable
        if `realized`, we make sure it is a vector, upcasting if necessary.
        """
        eta2 = self.eta2
        if isscalar(eta2):
            D = self.get_dim()

        if not realized:
            return eta2
        eta2 = torch.as_tensor(eta2)
        if eta2.dim() == 0:  # eta2 is a scalar tensor
            eta2 = eta2.expand(D)
        elif eta2.dim() == 1 and eta2.shape[0] == D:
            # eta2 is a vector of length D
            pass  # do nothing
        else:
            raise ValueError("Input eta2 is not compatible")
        return eta2

    def add_factor_node(self, factor_name, factor_node):
        self.factor_nodes[factor_name] = factor_node

    def observe(self, obs):
        """
        This node no longer participates in message-passing; it merely
        conditions adjacent factors.
        """
        self.observation = obs

    def is_observed(self):
        """
        return whether the node is observed
        """
        return self.observation is not None

    def get_observation(self):
        """
        return whether the node is observed
        """
        return self.observation

    def get_dim(self):
        """
        return the dimension of the variable
        """
        return self.ens.shape[1]

    def get_ens_size(self):
        """
        return the size of the ensemble
        """
        if self.ens is not None:
            return self.ens.shape[0]
        else:
            return self.n_ens

    def get_belief_rank(self):
        """
        return the factor rank of the belief
        """
        return self.belief[1].lr_factor.shape[0]

    def set_ens(self, ens):
        """
        update the current ensemble
        """
        self.ens = ens

    def get_ens(self, observed_override=False):
        """
        Return the current ensemble, which is just a matrix.

        observed_override makes a tiled copy of the observation; this is useful
        for when one input/parent to a system is observed, and we still want
        its influence to broadcast through.

        Alternatively, if we do not need the tiling, we can use
        `get_ens_or_obs`.
        """
        if self.ens is None:
            raise ValueError("No ensemble")
        if self.is_observed() and observed_override:
            return self.observation.unsqueeze(0).repeat(
                self.get_ens_size(), 1)
        else:
            return self.ens

    def get_ens_or_obs(self, ):
        """
        Return the current ensemble, which is just a matrix, or if observed,
        the observation, which is a vector

        Alternatively, if we need the obs to be cast to ensemble, try
        `get_ens` with `observed_override=True`.
        """
        if self.is_observed():
            return self.observation
        else:
            return self.ens

    # def compute_ens_moments(self):
    #     """
    #     calc the current ensemble moments.
    #     cache it for use by factor nodes.
    #     """
    #     self.ens_moments = moments_from_ens(
    #         self.get_ens(),
    #         self.get_gamma2(),
    #     )

    def send_messages(self, *args, **kwargs):
        """
        Compute the messages to the factor nodes and deliver to factor inboxes.
        Damping and bparams is ignored for this method
        """
        for k, f in self.factor_nodes.items():
            # print("v->f", k)
            # each factor gets a message made from all other messages except
            # for the one it sent to me
            temp_inbox = self.inbox.copy()
            if k in temp_inbox:
                temp_inbox.pop(k)
            if len(temp_inbox) > 0:
                m = variable_product_message(
                    list(temp_inbox.values()))
                ## DEBUG
                # prec = m[1]
                # det = torch.linalg.det(prec.to_tensor())
                # if det < -1e-6:
                #     raise ValueError(
                #         f"message prec is negative {self.name} to {f} ({det})")
                assert m[0].shape[0] == self.get_dim()
                ## END DEBUG
                f.inbox[self.name] = m

    def compute_message_belief(self):
        """
        Compute the marginal "belief" for this variable node, multiplying the
        messages from the inbox.

        This calculation is notionally evanescent;
        we do not use it in the message passing calculation and it do not
        (should not) change anything about what is propagated.
        But we cache it here for convergence diagnosis and model evaluation.

        TODO: iterative addition is not efficient for large inboxes.
        We should batch it.
        """
        es = [v[0] for v in self.inbox.values()]
        precs = [v[1] for v in self.inbox.values()]
        new_e = torch.sum(torch.stack(es), dim=0)
        new_prec = precs.pop(0)
        for p in precs:
            new_prec = new_prec + p
        # if not new_prec.lr_matrix.is_actually_lr():
        #     raise ValueError("message prec is not tall")
        return (new_e, new_prec)

    def update_message_canonical_belief(self):
        """
        Update the belief for this variable node.
        """
        self.belief = self.compute_message_belief()

    # def update_message_moments_belief(self):
    #     """
    #     Update the belief for this variable node from the canonical form.
    #     We need to call this if we want to estimate a model energy without
    #     ground truth, because this is how we find var means
    #     """
    #     if self.belief is None:
    #         warnings.warn(f"no canonical belief for {self.name}")
    #         return
    #     self.moments_belief = self.calc_message_moments_belief()

    # def calc_message_moments_belief(self):
    #     """
    #     What is my belief in intuitive moments form?
    #     """
    #     return moments_from_canonical(*self.belief)

    def get_belief(self):
        """
        What is my belief in canonical form?
        """
        return self.belief

    def get_moments_belief(self):
        """
        What is my belief in intuitive moments form?
        """
        return moments_from_canonical(*self.belief)

    def get_message_or_ens_belief(self):
        """
        What is my belief in canonical form?
        """
        if self.belief is not None:
            return self.belief
        else:
            return canonical_from_ens(self.get_ens(), self.get_gamma2())

    def get_message_or_ens_moments_belief(self):
        """
        What is my belief in intuitive moments form?

        What if we are trying to estimate a model energy without ground truth?
        We need to seed this with the ensemble belief
        """
        if self.moments_belief is not None:
            return self.get_moments_belief()
        else:
            return moments_from_ens(
                self.get_ens(), self.get_gamma2())

    def get_residual(self, obs):
        """
        residual
        """
        mean, var = self.get_moments_belief()
        return obs - mean

    def get_mse(self, obs):
        """
        residual wrt belief
        """
        residual = self.get_residual(obs)
        return (residual**2).mean()

    def get_belief_energy(self, obs):
        """
        residual energy wrt belief.

        Since elsewhere in the code this apparently elementary calculation
        produces complete nonsense, I have low faith in this bit.
        """
        e, prec = self.get_canonical_belief()
        e_obs = prec @ obs
        return 0.5 * (
            e_obs.reshape(1,-1)
            @ prec @ e_obs.reshape(-1, 1)
        ).squeeze() - (e_obs * e).sum()

    def get_ens_energy(self, obs):
        """
        residual energy wrt ensemble
        """
        mean, var = moments_from_ens(
            self.get_ens(),
            self.get_gamma2(),
        )
        residual = obs - mean
        return 0.5 * (
            residual.reshape(1,-1)
            @ var.solve(residual.reshape(-1, 1))
        ).squeeze()

    def get_loglik(self, obs):
        """
        log likelihood of obs wrt belief
        """
        mean, var = self.get_moments_belief()
        return LowRankMultivariateNormal(
            mean,
            var.lr_matrix.lr_factor,
            var.diag_matrix.diag_t()
        ).log_prob(obs)

    # def get_moments_ens_potential(self):
    #     """
    #     What are the moments implied by my ensemble?
    #     """
    #     return moments_from_ens(self.get_ens(), self.gamma2)

    def empty_inbox(self):
        """
        Empties belief inbox.
        """
        self.inbox.clear()

    def conform_ensemble(self, method="lstsq", **kwargs):
        """
        conform the ensemble to the belief
        """
        # print("where is my belief", self.name, self.belief)
        new_mean, new_dev = moments_from_canonical(*self.belief)
        ## DEBUG
        DEBUG_MODE = self.is_debug()

        if DEBUG_MODE:
            from matplotlib import pyplot as plt
            from matplotlib.lines import Line2D
            legend_handles = []
            legend_labels = []

            ens = self.get_ens()
            n_ens = ens.shape[0]
            full_D = ens.shape[1]
            alpha = n_ens ** -0.75
            # if I actually plot here the lines are invisible because a new
            # graph is created in conform_ensemble in debug mode
        ## END DEBUG

        self.ens, _residual, _energy_delta = conform_ensemble(
            method,
            self.ens,
            new_mean, new_dev,
            eta2=self.get_eta2(),
            DEBUG_MODE=DEBUG_MODE,
            **kwargs)

        ## DEBUG
        if DEBUG_MODE:
            for line_data in ens:
                plt.plot(
                    torch.arange(full_D),
                    line_data, color="red", alpha=alpha)
            legend_handles.append(Line2D([0], [0], color="red", lw=1))
            legend_labels.append('Prior ensemble')

            belief_ens = fake_ens_from_canonical(
                *self.belief, n_ens=n_ens)

            for line_data in belief_ens:
                plt.plot(
                    torch.arange(full_D),
                    line_data, color="green", alpha=alpha)
            legend_handles.append(Line2D([0], [0], color="green", lw=1))
            legend_labels.append('Target ensemble')
            for line_data in self.get_ens():
                plt.plot(
                        torch.arange(full_D),
                        line_data, color="blue", alpha=alpha)
            legend_handles.append(Line2D([0], [0], color="blue", lw=1))
            legend_labels.append(f'Posterior ensemble')

            plt.legend(legend_handles, legend_labels)
            plt.title(f"ENSEMBLE {self.name}: Conformed to belief")
            plt.show()
        ## END DEBUG
        # self.sigma2 = self.sigma2 #+ diag_delta
        return _energy_delta
