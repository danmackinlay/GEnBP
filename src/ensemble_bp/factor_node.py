import warnings
import torch

import numpy as np

from ..hermitian_matrices import DiagonalPlusLowRank
from ..torch_formatting import ensemble_packery
from ..gaussian_statistics import mean_dev_from_ens, \
    moments_from_ens, \
    moments_from_canonical, \
    canonical_from_moments, canonical_from_ens, \
    fake_ens_from_moments, fake_ens_from_canonical
from ._base import VERY_BIG_RANK, _factor_name, Converged
from .condition import pathwise_condition


class FactorNode:
    """
    A factor node in the factor graph.

    We assume that the factor node is a function of its parent nodes, and
    require the parent nodes to be populated.
    Child nodes' ensembles will be overwritten.
    Child status only matters during ancestral sampling;
    the rest of the time vars are exchangeable.
    """
    def __init__(
            self, sim_fn,
            parent_nodes={},
            child_nodes={},
            name="",
            fg=None):
        self.sim_fn = sim_fn
        self.parent_nodes = {}
        self.child_nodes = {}
        # all nodes we know about
        self.var_nodes = {}
        self.inbox = {}
        self.name = name
        self.ens_potential = None
        self.fg = fg
        for k, v in parent_nodes.items():
            self.add_parent_node(k, v)

        for k, v in child_nodes.items():
            self.add_child_node(k, v)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{_factor_name(self.sim_fn, self.parent_nodes, self.child_nodes)},"
            f"{self.parent_nodes.keys()},"
            f"{self.child_nodes.keys()},"
            ")"
        )

    def set_fg(self, fg):
        self.fg = fg

    def is_debug(self):
        return getattr(self.fg, 'DEBUG_MODE', False)

    def diagnosis(self):
        """
        Return a dict of diagnostic information about the factor node.
        """
        inbox_beliefs = {}
        for k, bel in self.inbox.items():
            e, prec = bel
            inbox_beliefs[k] = dict(
                e_shape=tuple(e.shape),
                prec=prec.diagnosis(),
                diag_mean=e.diag().mean().item(),
            )
        return dict(inbox_beliefs=inbox_beliefs)

    # setup stuff
    def add_parent_node(self, name, node):
        """
        Add a parent node to this factor node.
        """
        self.parent_nodes[name] = node
        self.var_nodes[name] = node

    def add_child_node(self, name, node):
        """
        Add a parent node to this factor node.
        """
        self.child_nodes[name] = node
        self.var_nodes[name] = node

    def get_var_nodes_d(self, include_observed=False):
        if include_observed:
            return self.var_nodes
        return {
            k: v
            for k, v in self.var_nodes.items()
            if not v.is_observed()}

    def get_parent_nodes(self, include_observed=False):
        if include_observed:
            return self.parent_nodes
        return {
            k: v
            for k, v in self.parent_nodes.items()
            if not v.is_observed()}

    def get_child_nodes(self, include_observed=False):
        if include_observed:
            return self.child_nodes
        return {
            k: v for k, v in self.child_nodes.items()
            if not v.is_observed()
        }

    # ensemble stuff
    def ancestral_sample(self):
        """
        Sample the prior ensemble for this factor node,
        by calculating the prior ensemble for each parent node,
        and then applying the prediction model.

        This happens BEFORE observing updates are applied,
        and is conceptually a linearisation step.
        We need to apply any observation updates again
        after each linearisation.
        """
        new_samples = self.sim_fn(
            *[
                p.get_ens_or_obs()
                for p in self.parent_nodes.values()
            ]
        )
        for k, v in zip(self.child_nodes, new_samples):
            self.child_nodes[k].set_ens(v)

    def get_observations_d(self):
        return {
            k: v.observation
            for k, v in self.var_nodes.items()
            if v.is_observed()}

    def get_observed_nodes_d(self):
        """
        Which nodes are observed
        """
        return {
            k: v for k, v in self.var_nodes.items() if v.is_observed()}

    def condition_on_observations(self):
        """
        We need to repeat this step each time the prior ensemble is updated.
        """
        ## DEBUG
        DEBUG_MODE = self.is_debug()
        if DEBUG_MODE:
            from matplotlib import pyplot as plt
            prior_ens = self.get_ens(include_observed=False).clone()
        ## END DEBUG
        observations = self.get_observations_d()
        if not observations:
            return
        sigma2s = {
            k: self.var_nodes[k].sigma2
            for k in self.var_nodes
        }
        upd = pathwise_condition(
            self.get_ens_d(include_observed=True),
            observations,
            sigma2s=sigma2s)
        ## DEBUG
        if DEBUG_MODE:
            print(f"conditioning {self.name} on {list(observations.keys())}")
            plt.figure()
        ## END DEBUG
        for k, v in upd[0].items():
            ## DEBUG
            if DEBUG_MODE:
                plt.plot(self.var_nodes[k].get_ens().T, alpha=0.1, color='red')
            ## END DEBUG
            self.var_nodes[k].set_ens(v)
            ## DEBUG
            if DEBUG_MODE:
                plt.plot(self.var_nodes[k].get_ens().T, alpha=0.1, color='blue')
                plt.title(f"ENSEMBLE {self.name} conditioning {k} from {list(observations.keys())}")
                plt.show()
            ## END DEBUG
        ## DEBUG
        # # None of my graphs have that many incoming obs for measurement nodes
        # # So let's ignore this plot for now;
        # # it will usually duplicate the previous one in that case.
        # if DEBUG_MODE:
        #     plt.figure()
        #     plt.plot(prior_ens.T, alpha=0.1, color='red')
        #     plt.plot(self.get_ens(include_observed=False).T, alpha=0.1, color='blue')
        #     plt.title(f"{self.name} conditioning")
        #     plt.show()
        ## END DEBUG

    def update_ens_potential(self, **kwargs):
        """
        Make potential consistent with the current ensemble.
        Be careful *when* this is done.
        We conditional all factors after ancestral sampling, and re-do observed factors after conditioning.
        """
        ## DEBUG
        DEBUG_MODE = self.is_debug()
        if DEBUG_MODE:
            from matplotlib import pyplot as plt
            if self.ens_potential is not None:
                message_ens = fake_ens_from_canonical(
                    *self.ens_potential, n_ens=100)
                col = "pink"
            else:
                message_ens = self.get_ens(include_observed=False)
                col = "red"
            for line_data in message_ens:
                plt.plot(
                    line_data, color=col, alpha=0.2)
        ## END DEBUG
        e, prec = canonical_from_ens(
            self.get_ens(include_observed=False),
            self.get_gamma2(),
            **kwargs)
        if torch.isnan(prec.lr_factor_t()).any():
            raise ValueError("nan in prec")

        assert len(e.shape) == 1
        assert e.shape[0] == self.get_dim()
        self.ens_potential = e, prec
        ## DEBUG
        if DEBUG_MODE:
            message_ens = fake_ens_from_canonical(
                *self.ens_potential, n_ens=100)
            for line_data in message_ens:
                plt.plot(
                    line_data, color='blue', alpha=0.2)
            plt.title(f"POTENTIAL {self.name} updated from ENSEMBLE")
            plt.show()
        ## DEBUG

    def get_ens_d(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.

        TODO: optionally fallback to a factor-local ensemble.
        """
        return self.get_ens_d_from_vars(
            include_observed=include_observed,
            observed_override=observed_override)

    def get_ens(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble matrix

        TODO: optionally fallback to a factor-local ensemble.
        """
        return self.get_ens_from_vars(
            include_observed=include_observed,
            observed_override=observed_override)

    def get_ens_or_obs_d(
            self):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.

        TODO: optionally fallback to a factor-local ensemble.
        """
        return self.get_ens_or_obs_d_from_vars()

    def get_ens_d_from_vars(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_ens(observed_override=observed_override)
            for k, v in var_nodes.items()
        }

    def get_ens_from_vars(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble matrix
        """
        return torch.cat([
            v for v in self.get_ens_d(
                include_observed=include_observed,
                observed_override=observed_override
            ).values()
        ], dim=1)

    def get_ens_or_obs_d_from_vars(
            self):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.
        """
        var_nodes = self.get_var_nodes_d(include_observed=True)
        return {
            k: v.get_ens_or_obs()
            for k, v in var_nodes.items()
        }

    def get_ens_d_from_factor(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.
        """
        raise NotImplementedError("TODO")
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_ens(observed_override=observed_override)
            for k, v in var_nodes.items()
        }

    def get_ens_from_factor(
            self,
            include_observed=False,
            observed_override=False):
        """
        return the current ensemble matrix
        """
        raise NotImplementedError("TODO")
        return torch.cat([
            v for v in self.get_ens_d(
                include_observed=include_observed,
                observed_override=observed_override
            ).values()
        ], dim=1)

    def get_ens_or_obs_d_from_factor(
            self):
        """
        return the current ensemble as a dict of matrices corresponding
        to var ensembles.
        """
        raise NotImplementedError("TODO")
        var_nodes = self.get_var_nodes_d(include_observed=True)
        return {
            k: v.get_ens_or_obs()
            for k, v in var_nodes.items()
        }

    def get_sigma2_d(self, include_observed=False, realized=False):
        """
        return the current sigma2 as a dict of scalars OR vectors
        corresponding to var nodes.
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_sigma2(realized=realized) for k, v in var_nodes.items()
        }

    def get_sigma2(self, include_observed=False):
        """
        sigma2 diagonal matrix
        """
        sigma2_d = self.get_sigma2_d(
            include_observed=include_observed, realized=True)
        return torch.cat(tuple(sigma2_d.values()), dim=0)

    def get_gamma2_d(self, include_observed=False, realized=False):
        """
        return the current gamma2 as a dict of scalars OR vectors
        corresponding to var nodes
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_gamma2(realized=realized) for k, v in var_nodes.items()
        }

    def get_gamma2(self, include_observed=False):
        """
        gamma2 diagonal, realised as a vector.
        Unused.
        """
        gamma2_d = self.get_gamma2_d(
            include_observed=include_observed, realized=True)
        return torch.cat(tuple(gamma2_d.values()), dim=0)

    def get_eta2_d(self, include_observed=False, realized=False):
        """
        return the current eta2 as a dict of scalars OR vectors
        corresponding to var nodes
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_eta2(realized=realized) for k, v in var_nodes.items()
        }

    def get_eta2(self, include_observed=False):
        """
        eta2 diagonal, realised as a vector.
        """
        eta2_d = self.get_eta2_d(
            include_observed=include_observed, realized=True)
        return torch.cat(tuple(eta2_d.values()), dim=0)

    def is_observed(self):
        """
        Is this factor node observed?
        """
        return len(self.get_observations_d()) > 0

    def get_var_dims_d(self, include_observed=False):
        """
        return the dimensions of each implicated var node
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return var_dims_d(var_nodes)

    def get_var_slices_d(self, include_observed=False):
        """
        return slice slices for each var node
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return var_slices_d(var_nodes)

    def slice_by_var_d(self, mat, include_observed=False):
        """
        Returns a dict of views of a large matrix with slices
        corresponding to the var nodes.
        This should turn a matrix into a dict of matrices.

        AFAIK this is not used even though it would be helpful.
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return slice_by_var_d(var_nodes, mat)

    def get_dim(self, include_observed=False):
        return sum([
            v.get_dim() for v in self.get_var_nodes_d(
                include_observed=include_observed).values()
        ])

    def get_belief(self):
        """
        Return the current belief potential as a 2-tuple of (e, prec.
        We calculate it by finding a product of the inbox messages and the factor potential.
        """
        raise NotImplementedError("TODO")
        return self.ens_potential

    def get_belief_energy(self, eval_d={}):
        """Computes the squared error for this factor using the belief potential weighting
        """
        raise NotImplementedError("Currently broken; we use an empirical approximation instead")
        # Weighting from factor belief  potential
        factor_belief =  self.get_belief()
        e, prec = factor_belief
        # estimate falls back to \*variable\* mean
        var_nodes = self.get_var_nodes_d(include_observed=False)
        eval_d_actual = {}
        for k, var_node in var_nodes.items():
            if eval_d.get(k, None) is None:
                eval_d_actual[k] = var_node.get_message_or_ens_moments_belief()[0]
            else:
                eval_d_actual[k] = eval_d[k]

        eval_v = torch.cat([mean for mean in eval_d_actual.values()], dim=0)

        # here is the un-normalised Gaussian potential, i.e. log density, via canonical form
        energy = (
            0.5 * eval_v[None, :] @ prec @ eval_v[:, None]
        ).squeeze() - (e * eval_v).sum()
        if energy < 0:
            raise ValueError(f"negative energy {energy}")
        return energy

    def get_ens_energy(self, eval_d={}):
        """Computes the weighted squared error from ensemble statistics

        NB if eval_d is not complete we use the variable means.
        """
        # Weighting from factor ensemble potential
        # estimate falls back to \*variable\* mean
        var_nodes = self.get_var_nodes_d(include_observed=False)
        eval_d_actual = {}
        for k, var_node in var_nodes.items():
            if eval_d.get(k, None) is None:
                eval_d_actual[k] = var_node.get_message_or_ens_moments_belief()[0]
            else:
                eval_d_actual[k] = eval_d[k]
        eval_v = torch.cat([mean for mean in eval_d_actual.values()], dim=0)
        # self.ens_potential should contain the updated energy because we just called update_ens_potential via self.fg.update_factor_conditional_ens_potential
        #it does not. I don't know why. Let's calculate it again I guess
        # return self._get_energy(eval_d, self.ens_potential)
        mean, var = moments_from_ens(
            self.get_ens(include_observed=False),
            self.get_gamma2()
        )
        # Now find the un-normalised Gaussian potential, i.e. log density, via canonical form
        # this result ends up being total bullshit and is not even positive
        # e, prec = canonical_from_ens(
        #     self.get_ens(include_observed=False),
        #     self.get_gamma2()
        # )
        # energy = (
        #     0.5 * eval_v[None, :] @ prec @ eval_v[:, None]
        # ).squeeze() - torch.dot(e, eval_v)
        # Let's try it in moments form, which should be equivalent
        residual = eval_v - mean
        energy = (
            0.5 * residual[None, :] @ var.inv() @ residual[:, None]
        ).squeeze()

        if energy < 0:
            raise ValueError(f"negative energy {energy}")
        return energy

    # def get_mean_dev_d(self, include_observed=False):
    #     """
    #     return the current ensemble as a dict of matrices corresponding to
    #     var ensembles.
    #     ]"""
    #     return {
    #         k: v.get_ens_mean_dev()
    #         for k, v in self.get_var_nodes_d(
    #             include_observed=include_observed
    #         ).items()
    #     }

    # def get_ens_mean_dev(self, include_observed=False):
    #     """
    #     calc the current ensemble mean/dev.
    #     """
    #     mean_dev_d = self.get_mean_dev_d(include_observed=include_observed)
    #     big_mean = torch.cat(
    #         tuple([md[0] for md in mean_dev_d.values()]),
    #         dim=0)
    #     big_dev = torch.cat(
    #         tuple([md[1] for md in mean_dev_d.values()]),
    #         dim=1)
    #     return big_mean, big_dev

    def empty_inbox(self):
        """
        Empties belief inbox.
        """
        self.inbox.clear()

    def send_messages(
            self, max_rank=None,
            damping=0.0,
            hard_damping=False,  # damp even first step
            **bparams):
        """
        Compute the messages to the variable nodes and put them in the correct
        inboxes.
        Most work happens in `_factor_to_var_messages`,
        where the idealised messages are calculated.
        Here we deal with transmitting the messages, which means damping and
        addressing them.

        Let us fix an interpretation of damping to match Ortiz.

        m_damped = (1-damping) * m_new + damping * m_old

        We need to be careful if m_old is null;
        in that case damping only applies if hard_damping is True, otherwise we
        go un-damped.
        """
        if max_rank is None:
            raise Exception("max rank panic")
        if damping >= 1.0 and hard_damping:
            raise ValueError(
                "damping=1.0 and hard_damping causes no updates ever")
        for k, m in self._factor_to_var_messages(
                max_rank=max_rank, **bparams).items():

            if damping == 0.0:
                # No damping; then we don't care about hard_damping
                # let message through unchanged
                pass
            elif damping > 0.0 and self.name in self.var_nodes[k].inbox:
                # There is an old message to work with.
                # Take the weighted sum.
                old_e, old_prec = self.var_nodes[k].inbox[self.name]
                new_e, new_prec = m
                m = (
                    (1-damping) * new_e + damping * old_e,
                    new_prec.mul(1-damping) + old_prec.mul(damping),
                )
                ## DEBUG
                # print(f"DAMPING {damping}")
                # pprint(old_prec.diagnosis())
                # pprint(new_prec.diagnosis())
                # pprint(m[1].diagnosis())
                ## END DEBUG
            elif damping > 0.0 and hard_damping:
                # fade it in
                new_e, new_prec = m
                m = (
                    (1-damping) * new_e,
                    new_prec.mul(1 - damping)
                )
                ## DEBUG
                # print(f"HARDDAMPING {damping}")
                # pprint(new_prec.diagnosis())
                # pprint((m[1].diagnosis()))
                ## END DEBUG
            else:
                # damping > 0.0 and no old message and not hard_damping
                # let message through unchanged
                pass
            ## DEBUG
            assert m[0].shape[0] == self.var_nodes[k].get_dim()
            ## END DEBUG
            self.var_nodes[k].inbox[self.name] = m

    def _factor_to_var_messages(
            self, max_rank=None, **bparams):
        """
        For *each* outgoing var, we need to construct a marginal canonical
        belief message made from concatenating all the other var nodes'
        messages times factor potential, then marginalising.
        """
        outbox = {}
        keys = self.get_var_nodes_d(include_observed=False).keys()
        for k in keys:
            # print("_f->v", k)
            outbox[k] = self._factor_to_var_message(
                k, max_rank=max_rank, **bparams)
        return outbox

    def _factor_to_var_message(
            self, target_var_key, max_rank=None, **bparams):
        """
        Create one factor-to-var message from the _other_ var nodes' messages.
        Messages are dicts of (e, prec) pairs.

        This method redirects to either a full marginalisation, or to a
        shortcut method if there is only one unobserved var in the factor.
        """
        var_nodes = self.get_var_nodes_d(include_observed=False)
        if len(var_nodes) == 1:
            if target_var_key not in var_nodes:
                raise ValueError("var key not in var nodes")
            return self._factor_to_var_message_leaf(target_var_key)
        else:
            return self._factor_to_var_message_full(
                target_var_key, max_rank=max_rank, **bparams)

    def _factor_to_var_message_leaf(self, target_var_key):
        """
        Create one factor-to-var message for an observation-conditional
        singleton factor.
        There is only one unobserved var in the factor, so we can do a
        shortcut marginalisation, simply returning our own belief.

        Is something weird here? It sounds like we can never update our
        outgoing message, because there are no other incoming messages to
        incorporate to update our inbox. Is that OK?
        """
        return self.ens_potential

    def _factor_to_var_message_full(
            self,
            target_var_key, max_rank=None, atol=0.0, rtol=None, retain_all=False, **bparams):
        """
        Create one factor-to-var message for `target_var_key` from the _other_
        var nodes' messages.

        Messages are dicts of (e, prec) pairs.

        TODO: if the inbox is empty, or has nothing in it that is not the
        target, we can just use the factor ensemble belief as the message,
        since we will not update.

        **Marginalised factor-to-var message recipe:**

        First we concatenate all the padded precision factors into a
        block-sparse precision factor, and add together the e terms. Then we
        invert and calculate moments form.

        Then we marginalise over the target node, by ignoring unneeded
        dims of the moment-form mean and covariance.

        The final message is the canonical params calculated from the
        marginalised node.

        I suspect we could save compute here by calculating the messages
        en masse, caching, exploiting block sparsity  and using
        Schur complements or even Lanczos decomposition.
        :shrug:
        """
        if max_rank is None:
            max_rank = VERY_BIG_RANK

        messages = self.inbox.copy()
        var_nodes = self.get_var_nodes_d(include_observed=False)
        name = self.name
        factor_e, factor_prec = self.ens_potential

        if target_var_key in messages:
            del messages[target_var_key]
        # Some of this dimension-calc logic is duplicated elsewhere.
        # TODO: refactor
        ## DEBUG
        DEBUG_MODE = self.is_debug()
        dist_debug_plot = lambda *args, **kwargs: None
        if DEBUG_MODE:
            from matplotlib import pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib import cm

            PLOT = len(messages) > 0
            # PLOT = False
            print("-"*80)
            print(
                f"calculating factor_to_var_message {target_var_key} from {list(messages.keys())} for {name}")

            def dist_debug_plot(m1, m2, title, **hm_kwargs):
                if not PLOT: return
                # sd is only meaningful in moments form
                sd = m2.diag_t().sqrt()
                x = np.arange(len(m1))
                plt.clf()
                plt.plot(m1, label="m1")
                plt.fill_between(x, m1 - sd, m1 + sd, color='gray', alpha=0.5)
                plt.title(title)
                plt.show()
                plt.clf()
                m2.diagnostic_plot(title, **hm_kwargs)
                plt.show()
        ## END DEBUG

        # calculate the size of the big moment matrices
        # ambient dims
        target_D = var_nodes[target_var_key].get_dim()
        full_D = factor_e.shape[0]
        # we need to know the size of each var node
        D_size_d = {}
        D_slice_d = {}
        D_start_d = {}
        D_end_d = {}
        current_D = 0

        # TODO: instead, use `slice_by_var_d`,` `var_dims_d`` etc
        for k, v in var_nodes.items():
            D_start_d[k] = current_D
            D = v.get_dim()
            D_size_d[k] = D
            D_slice_d[k] = slice(current_D, current_D + D)
            current_D += D
            D_end_d[k] = current_D
            # no update from the target of the message
            if k == target_var_key:
                continue

        assert current_D == full_D
        ## DEBUG
        # print(f"factor dim {full_D} target dim {target_D}")
        # dist_debug_plot(factor_e, factor_prec, f"{name}: factor canonical")
        # dist_debug_plot(
        #     *moments_from_canonical(factor_e, factor_prec),
        #     f"{name}: factor moments normed", normalize=True)
        ## END DEBUG

        # create the big precision matrix
        big_e = factor_e.detach().clone()
        big_prec = factor_prec

        ## DEBUG
        if DEBUG_MODE and len(messages) > 0:
            plt.clf()
            n_clusters = len(messages) + 3
            colors = cm.viridis(np.linspace(0, 1, n_clusters))
            legend_handles = []
            legend_labels = []
            i = 0

            factor_ens = self.get_ens(include_observed=False)
            n_ens = factor_ens.shape[0]
            alpha = n_ens ** -0.5
            for line_data in factor_ens:
                plt.plot(
                    torch.arange(full_D),
                    line_data, color=colors[i], alpha=alpha)
            legend_handles.append(Line2D([0], [0], color=colors[i], lw=4))
            legend_labels.append('Empirical factor prior')
            i += 1
            belief_ens = fake_ens_from_canonical(
                big_e, big_prec, n_ens=n_ens)
            for line_data in belief_ens:
                plt.plot(
                    torch.arange(full_D),
                    line_data, color=colors[i], alpha=alpha)
            legend_labels.append('BP factor prior')
            legend_handles.append(Line2D([0], [0], color=colors[i], lw=4))

        ## END DEBUG

        for k, m in messages.items():
            e, prec = m
            before = D_start_d[k]
            after = full_D - D_end_d[k]
            ## DEBUG
            # _, message_var = moments_from_canonical(*m, **bparams)
            # pprint({k: message_var.diagnosis()})
            ## END DEBUG
            big_e[D_slice_d[k]] += e
            # TODO: this allocates memory, and does not exploit sparsity
            big_prec = big_prec + prec.padded(before, after)
            ## DEBUG
            if DEBUG_MODE:
                i += 1
                message_ens = fake_ens_from_canonical(
                    e, prec, n_ens=n_ens)
                for line_data in message_ens:
                    plt.plot(
                        torch.arange(start=D_start_d[k], end=D_end_d[k]),
                        line_data, color=colors[i], alpha=alpha)
                legend_handles.append(Line2D([0], [0], color=colors[i], lw=4))
                legend_labels.append(f'message {k}')
            # dist_debug_plot(
            #     e, prec,
            #     f"{name}: factor canonical update from {k}")
            # dist_debug_plot(
            #     *moments_from_canonical(e, prec),
            #     f"{name}: factor moments updated from {k} normed",
            #     normalize=True)
            # dist_debug_plot(
            #     big_e, big_prec,
            #     f"{name}: factor canonical after {k}")
            # dist_debug_plot(
            #     *moments_from_canonical(big_e, big_prec),
            #     f"{name}: factor moments after {k} normed",
            #     normalize=True)
            ## END DEBUG

        ## DEBUG
        # how much did we update in canonical space?
        # delta_prec = big_prec.as_dense() + factor_prec.as_dense().mul(-1)
        # delta_e = big_e - factor_e
        # dist_debug_plot(
        #     delta_e, delta_prec,
        #     f"{name}: factor delta canonical")
        # dist_debug_plot(
        #     *moments_from_canonical(delta_e, delta_prec),
        #     f"{name}: factor delta moments")
        ## END DEBUG
        # We can marginalise over any node by finding the moments
        # then slicing out the relevant indices
        big_m, big_var = moments_from_canonical(
            big_e, big_prec, **bparams)
        ## DEBUG
        if DEBUG_MODE and len(messages) > 0:
            posterior_factor_ens = fake_ens_from_moments(
                big_m, big_var, n_ens=n_ens)
            i += 1
            for line_data in posterior_factor_ens:
                plt.plot(
                    torch.arange(full_D),
                    line_data, color=colors[i], alpha=alpha)
            legend_handles.append(Line2D([0], [0], color=colors[i], lw=4))
            legend_labels.append(f'factor posterior')

            plt.legend(legend_handles, legend_labels)
            plt.title(f"POTENTIAL {name}: factor update updated from MESSAGES {list(messages.keys())}")
            plt.show()
        ## END DEBUG
        assert len(big_m.shape) == 1
        assert big_m.shape[0] == full_D
        ## DEBUG
        if DEBUG_MODE:
            pprint({"big_var": big_var.diagnosis()})
            # check for nans in factors
            if torch.isnan(big_prec.lr_factor_t()).any():
                raise ValueError("nan in big_prec")
            if torch.isnan(big_var.lr_factor_t()).any():
                raise ValueError("nan in big_var")
        ## DEBUG
        # dist_debug_plot(big_m, big_var, f"{name}: big moments")
        # how much did we update in moments space?
        # prior_m, prior_var = moments_from_ens(
        #     self.get_ens(include_observed=False),
        #     self.get_gamma2(),
        # )
        # delta_m = prior_m - big_m
        # delta_var = prior_var.as_dense() + big_var.as_dense().mul(-1)
        # dist_debug_plot(
        #     delta_m, delta_var,
        #     f"{name}: factor delta update moments")
        # dist_debug_plot(
        #     delta_m, delta_var,
        #     f"{name}: factor delta update normed",
        #     normalize=True)
        ## END DEBUG

        # Control rate of growth of rank while in moments form, when it has a
        # meaningful interpretation
        big_var = big_var.reduced_rank(max_rank, atol=atol, rtol=rtol)
        dist_debug_plot(big_m, big_var, f"{name}: big moments reduced rank")
        if big_var.rank() > max_rank:
            raise Exception(
                f"how is rank {big_var.rank()} still bigger than {max_rank}?")
        # now, marginalize over the target node
        marginal_m = big_m[D_slice_d[target_var_key]]
        marginal_var = big_var.extract_block(
            D_start_d[target_var_key], D_end_d[target_var_key])

        ## DEBUG
        # dist_debug_plot(
        #     marginal_m, marginal_var, f"{name}/{target_var_key}: marginal moments")
        # dist_debug_plot(
        #     marginal_m, marginal_var,
        #     f"{name}/{target_var_key}: marginal moments normed",
        #     normalize=True)
        ## END DEBUG
        message = canonical_from_moments(marginal_m, marginal_var, **bparams)
        ## DEBUG
        # dist_debug_plot(*message, f"{name}/{target_var_key}: marginal canonical")
        ## END DEBUG
        assert len(message[0].shape) == 1
        assert message[0].shape[0] == target_D

        return message
