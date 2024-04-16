import warnings
import torch

from ._base import VERY_BIG_RANK, _factor_name, Converged
from .var_node import VarNode
from .factor_node import FactorNode


class FactorGraph:
    """
    FactorGraph objects enforce (some parts of) a sane inference cycle,
    by setting up dicts of var and factor nodes all of whom are aware of each
    other.
    """

    @classmethod
    def from_sem(
            cls, sem,
            ancestor_samples,
            sigma2=0.0, sigma2s=None,
            gamma2=0.0, gamma2s=None,
            eta2=None, eta2s=None,
            n_ens=None,
            **kwargs):
        """
        Create a factor graph from a structural equation model.
        `sem` is a list of tuples each of the form
        (sim_fn, input_names, output_names).
        sem is assumed to be in topological order, i.e. ancestors first.
        """
        factor_nodes = {}
        var_nodes = {}
        if sigma2s is None:
            sigma2s = {}
        if gamma2s is None:
            gamma2s = {}
        if eta2s is None:
            eta2s = gamma2s
        if eta2 is None:
            eta2 = gamma2
        for (sim_fn, input_names, output_names) in sem:
            # create the factor nodes mentioned in this SEM
            for name in input_names + output_names:
                this_sigma2 = sigma2s.get(name, sigma2)
                this_gamma2 = gamma2s.get(name, gamma2)
                this_eta2 = eta2s.get(name, eta2)
                var_node = var_nodes.setdefault(
                    name, VarNode(
                        name, {},
                        sigma2=this_sigma2,
                        gamma2=this_gamma2,
                        eta2=this_eta2,
                    ))
            factor_name = _factor_name(
                sim_fn, input_names, output_names)
            parent_nodes = {name: var_nodes[name] for name in input_names}
            child_nodes = {name: var_nodes[name] for name in output_names}
            factor_node = FactorNode(
                sim_fn, parent_nodes, child_nodes, name=factor_name,
            )
            factor_nodes[factor_name] = factor_node
            # tell all the variable nodes about this factor node
            for name in input_names + output_names:
                var_nodes[name].add_factor_node(
                    factor_name, factor_node)
        for name, var_node in var_nodes.items():
            if name in ancestor_samples:
                var_node.set_ens(ancestor_samples[name])
                n_ens = ancestor_samples[name].shape[0]
        if n_ens is None:
            raise ValueError("n_ens must be specified if there are no samples")
        return cls(
            factor_nodes, var_nodes,
            n_ens=n_ens,
            **kwargs)

    def __init__(
            self, factor_nodes, var_nodes,
            n_ens=None, **settings):
        """
        Construct the FG.
        Nodes are assumed to be in topological order, i.e. ancestors first.
        """
        self.n_ens = n_ens
        self.factor_nodes = factor_nodes
        self.var_nodes = var_nodes
        self._settings = settings
        self._settings.setdefault("max_rank", VERY_BIG_RANK)
        self._settings.setdefault("cvg_tol", -1.0)
        self._settings.setdefault("damping", 0.0)
        self._settings.setdefault("max_steps", 50)
        self._settings.setdefault("min_mp_steps", 1)
        self._settings.setdefault("max_relin_int", 5)
        self._settings.setdefault("hard_damping", True)
        self._settings.setdefault("callback", lambda x: None)
        self._settings.setdefault("empty_inboxes", True)
        self._settings.setdefault("sigma2_scale", 1.0)
        self._settings.setdefault("gamma2_scale", 1.0)
        self._settings.setdefault("belief_retain_all", False)
        self._settings.setdefault("conform_retain_all", True)
        self._settings.setdefault("conform_r_eigen_floor", 1e-4)
        self._settings.setdefault("conform_randomize", True)
        self._settings.setdefault("conform_method", "lstsq")
        self._settings.setdefault("conform_between", True)
        self._settings.setdefault("verbose", 0)
        self._settings.setdefault("atol", 0.)
        self._settings.setdefault("rtol", None)
        self._settings.setdefault("schedule", "all")
        self._settings.setdefault("DEBUG_MODE", False)

        for factor_node in self.factor_nodes.values():
            factor_node.set_fg(self)
        for var_node in self.var_nodes.values():
            var_node.set_fg(self)

    def __repr__(self):
        return f"{type(self).__name__}({self.factor_nodes}, {self.var_nodes})"

    def is_debug(self):
        return self.get_setting('DEBUG_MODE', False)

    def diagnosis(self):
        """
        Return a dict of diagnostic information about the graph.
        """
        return dict(
            var_nodes={
                k: v.diagnosis()
                for k, v in self.var_nodes.items()},
            factor_nodes={
                k: f.diagnosis()
                for k, f in self.factor_nodes.items()},
        )

    def ancestral_sample(self):
        """
        Sample from the prior.
        """
        for factor_name, factor_node in self.factor_nodes.items():
            # print("sampling", factor_name)
            factor_node.ancestral_sample()

    def get_var_node(self, name):
        """
        Get a variable node by name.
        """
        return self.var_nodes[name]

    def get_factor_nodes_d(self):
        """
        Get a factor node dict; I thought I needed this helper for filtering by
        observed, but maybe not?
        """
        return self.factor_nodes

    def get_factor_node(self, name):
        """
        Get a factor node by name.
        """
        return self.factor_nodes[name]

    def get_all_factors_for_d(self, names):
        """
        Get all factor nodes implicating these vars by name
        """

        def yep(factor):
            return all([name in factor.var_nodes for name in names])

        return {k: v for k, v in self.factor_nodes.items() if yep(v)}

    def get_factor_for(self, names):
        """
        Get all factor nodes implicating exactly these vars by name
        """
        names = set(names)

        def yep(factor):
            return names == factor.var_nodes.keys()

        candidates = [v for v in self.factor_nodes.values() if yep(v)]
        if len(candidates) == 0:
            raise KeyError(f"No factor node with vars {names}")
        elif len(candidates) > 1:
            raise KeyError(f"Multiple factor nodes with vars {names}")
        return candidates[0]

    # def get_all_nodes(self):
    #     """
    #     Get all nodes. We never need to do this.
    #     """
    #     return {
    #         **self.var_nodes,
    #         **self.factor_nodes,
    #     }

    def get_ancestor_vars(self, include_observed=False):
        """
        Which nodes must I resample to do ancestral sampling
        """
        # descendants = {}
        # If a var node is not a child wrt any factor node, it is an ancestor
        ancestors = self.get_var_nodes_d(
            include_observed=include_observed)
        for factor_name, factor_node in self.factor_nodes.items():
            for var_name, var_node in factor_node.get_child_nodes(
                    include_observed=True).items():
                if var_name in ancestors:
                    del ancestors[var_name]
                # descendants[var_name] = var_node
        return ancestors

    def get_ancestor_factors(self, include_observed=False):
        """
        Which factors must I resample to do ancestral sampling
        """
        # descendants = {}
        # If a var node is not a child wrt any factor node, it is an ancestor
        # ancestor_vars = self.get_ancestor_vars(
        #     include_observed=include_observed)
        raise NotImplementedError

    def get_var_nodes_d(self, include_observed=False):
        if include_observed:
            return self.var_nodes.copy()
        else:
            return {
                k: v
                for k, v in self.var_nodes.items()
                if not v.is_observed()}

    def set_ens_d(self, ens):
        """
        return the current ensemble
        """
        raise NotImplementedError()

    def get_ens_d(self, include_observed=False):
        """
        return the current ensemble divided up into var nodes
        """
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_ens() for k, v in var_nodes.items()
        }

    def get_ens(self, include_observed=False):
        """
        return the current ensemble as a matrix
        """
        ens_d = self.get_ens_d(include_observed=include_observed)
        return torch.cat([v for v in ens_d.values()], dim=1)

    def get_residual_d(self, obs_d):
        """
        residual over a whole factor.
        """
        var_nodes = self.get_var_nodes_d(include_observed=False)
        return {
            k: var_node.get_residual(obs_d.get(k, None))
            for k, var_node in var_nodes.items()
        }

    def get_mean_d(self, include_observed=False):
        var_nodes = self.get_var_nodes_d(include_observed=include_observed)
        return {
            k: v.get_moments_belief()[0] for k, v in var_nodes.items()
        }

    def get_ens_energy_d(self, eval_d={}):
        """ Computes the squared error using the weighted mse, per factor

        If obs_d is not complete we use the var means.
        """
        return {
            k: f.get_ens_energy(eval_d) for k, f in self.get_factor_nodes_d().items()
        }

    def get_ens_energy(self, eval_d={}):
        """ Computes the total squared error using the weighted mse.

        NB if obs_d is not complete we use the var means.
        """
        energy_d = self.get_ens_energy_d(eval_d=eval_d)
        return torch.sum(torch.stack([v for v in energy_d.values()]))

    def get_belief_energy_d(self, eval_d={}):
        """ Computes the squared error using the weighted mse, per factor

        If obs_d is not complete we use the var means.
        """
        return {
            k: f.get_belief_energy(eval_d) for k, f in self.get_factor_nodes_d().items()
        }

    def get_belief_energy(self, eval_d={}):
        """ Computes the total squared error using the weighted mse.

        NB if obs_d is not complete we use the var means.
        """
        energy_d = self.get_belief_energy_d(eval_d=eval_d)
        return torch.sum(torch.stack([v for v in energy_d.values()]))

    # def get_belief_d(self, include_observed=False):
    #     var_nodes = self.get_var_nodes_d(include_observed=include_observed)
    #     return {
    #         k: v.get_belief() for k, v in var_nodes.items()
    #     }

    # def get_moments_belief_d(self, include_observed=False):
    #     var_nodes = self.get_var_nodes_d(include_observed=include_observed)
    #     return {
    #         k: v.get_moments_belief() for k, v in var_nodes.items()
    #     }

    # def get_moments_ens_potential_d(self, include_observed=False):
    #     var_nodes = self.get_var_nodes_d(include_observed=include_observed)
    #     return {
    #         k: v.get_moments_ens_potential() for k, v in var_nodes.items()
    #     }

    def observe_d(self, observations):
        """
        Observe the factor nodes, by conditioning on the observed variables.
        """
        for k, v in observations.items():
            self.observe(k, v)

    def observe(self, k, val):
        """
        Observe the factor node, by conditioning on the observed variable.
        """
        var_node = self.var_nodes[k]
        var_node.observe(val)

    def condition_on_observations(self):
        for factor_name, factor_node in self.factor_nodes.items():
            factor_node.condition_on_observations()

    def get_observed_nodes_d(self):
        """
        Which nodes are observed
        """
        return {k: v for k, v in self.var_nodes.items() if v.is_observed()}

    def get_observations_d(self):
        """
        What are the current observations
        """
        return {
            k: v.observation for k, v in self.get_observed_nodes()
        }

    def update_factor_ens_potential(self, **kwargs):
        """
        Construct beliefs that are consistent with the prior ensemble.
        """
        for factor_name, factor_node in self.factor_nodes.items():
            factor_node.update_ens_potential(**kwargs)

    def update_factor_conditional_ens_potential(self, **kwargs):
        """
        Construct beliefs that are consistent with the current ensemble at all
        conditioned, observed nodes.
        """
        for factor_name, factor_node in self.factor_nodes.items():
            if factor_node.is_observed():
                factor_node.update_ens_potential(**kwargs)

    # def update_var_ens_potential(self, **kwargs):
    #     """
    #     Construct beliefs that are consistent with the current ensemble;
    #     This is not necessary in for message passing but makes easy
    #     diagnostics.
    #     """
    #     for var_name, var_node in self.var_nodes.items():
    #         var_node.update_ens_potential(**kwargs)

    def update_message_canonical_belief(self):
        """
        Update the beliefs from the messages.
        """
        for var_node in self.get_var_nodes_d(
                include_observed=False).values():
            var_node.update_message_canonical_belief()

    def update_message_moments_belief(self):
        """
        Update the belief variable nodes FROM THE CANONICAL-FORM MESSAGES.
        We need to call this if we want to estimate a model energy without
        ground truth, because this ~~is~~ was how we find var means
        """
        raise NotImplementedError("no longer needed")
        for var_node in self.get_var_nodes_d(
                include_observed=False).values():
            var_node.update_message_moments_belief()

    def scale_sigma2(self, scale):
        for var_name, var_node in self.var_nodes.items():
            var_node.scale_sigma2(scale)

    def scale_gamma2(self, scale):
        for var_name, var_node in self.var_nodes.items():
            var_node.scale_gamma2(scale)

    def get_settings(self):
        """
        Get the settings for this factor graph inference
        """
        return self._settings

    def get_setting(self, key, *fallbackarg):
        """
        Get a setting for this factor graph inference
        """
        if len(fallbackarg) > 0:
            return self._settings.get(key, fallbackarg[0])
        return self._settings[key]

    def set_settings(self, **settings):
        """
        update the settings for this factor graph inference
        """
        self._settings.update(settings)

    def _update_belief_from_ensemble(self):
        self._steps_since_relin = 0
        self.ancestral_sample()
        self.update_factor_ens_potential()
        self.condition_on_observations()
        self.update_factor_conditional_ens_potential()
        # self.update_var_ens_potential()
        self.scale_sigma2(self.get_setting("sigma2_scale"))
        self.scale_gamma2(self.get_setting("gamma2_scale"))

    def _update_sample_from_belief(self):
        energy_delta = self.conform_ancestors()
        if self.get_setting("empty_inboxes"):
            self.empty_inboxes()
        return energy_delta

    def _mp_step(self):
        """
        A single iteration of message passing.
        broken into its own method so we can watch convergence interactively.
        """
        # BP iterations
        schedule = self.get_setting("schedule")
        if schedule == "all":
            self.send_messages_all()
        elif schedule == "random":
            self.send_messages_random()
        callback = self.get_setting("callback")
        callback(self)
        self.update_message_canonical_belief()
        ## no longer needed?
        # self.update_message_moments_belief()
        # TODO: calculate belief energy delta
        # self._prev_belief_energy = self._belief_energy
        # self._belief_energy = self.get_belief_energy()
        # belief_energy_delta = self._prev_belief_energy - self._belief_energy
        belief_energy_delta = 0.0
        self._n_mp_steps += 1
        self._steps_since_relin += 1
        return belief_energy_delta

    def solve(self, callback=lambda i, fg: None):
        """
        Complete inference loop.
        This is how we would usually invoke it;
        but we break out some work into methods so we can analyse the convergence.
        """
        self._n_mp_steps = 0
        self._ens_energies = []
        self._callback_log = []
        self._ens_energy = 1e8
        self._belief_energy = self._ens_energy
        cvg_tol = self.get_setting("cvg_tol")
        while self._n_mp_steps < self.get_setting("max_steps"):
            if self._n_mp_steps == 0:
                # "first" step; just update empirical potentials
                self._update_belief_from_ensemble()
            else:
                # We just relinearized. Check if we converged
                self._prev_ens_energy = self._ens_energy
                self._update_belief_from_ensemble()
                self._ens_energy = self.get_ens_energy()
                self._ens_energies.append(self._ens_energy)
                ens_energy_delta = self._ens_energy - self._prev_ens_energy
                # if energy didn't decrease by much, or even increased, we need to stop
                if (
                        ens_energy_delta > cvg_tol
                    ):
                    print(f"simulation converged with energy {self._ens_energy} since delta {ens_energy_delta}>={cvg_tol}")
                    if ens_energy_delta > 0.0:
                        warnings.warn(f"simulation diverged")
                    break

            while (self._steps_since_relin < self.get_setting("max_relin_int")
                    and self._n_mp_steps < self.get_setting("max_steps")):
                #TODO: actually calc belief energy delta, possible finish early
                belief_energy_delta = self._mp_step()
                # if self.is_debug():
                #     warnings.warn(f"belief_energy_delta={belief_energy_delta}")
                # if (
                #         belief_energy_delta < cvg_tol and
                #         self._steps_since_relin >= self.get_setting("min_mp_steps")  # off by 1?
                #     ):
                #     warnings.warn(f"mp converged {belief_energy_delta}<={cvg_tol}")
                #     break
            callback_rtn = callback(self._n_mp_steps, self)
            if callback_rtn is not None:
                self._callback_log.append(callback_rtn)

            warnings.warn(f"converged conforming after {self._steps_since_relin} steps")
            self._update_sample_from_belief()

        return self._ens_energies

    def conform_ancestors(self, include_observed=False):
        """
        Find a transform such that the ensemble has approximately the right
        moments.

        Currently only one ancestral node is supported. If we want to infer
        multiple latents *jointly*, one way is to concatenate them into a
        single var.

        TODO: This is a bug. conforming should happen across ancestral
        *factors*, which would be consistent across multiple vars jointly.

        workaround: create a special sim function taking all ancestors as
        input, and use that as factor.

        If this work ends up being useful, will update.
        """
        energy_delta = 0.
        ancestors = self.get_ancestor_vars(include_observed=include_observed)
        if len(ancestors) > 1:
            warnings.warn(
                f"Inference for ancestors {ancestors.keys()}"
                " may not be jointly valid")
        for ancestor in ancestors.values():
            energy_delta += ancestor.conform_ensemble(
                conform_r_eigen_floor=self.get_setting("conform_r_eigen_floor"),
                retain_all=self.get_setting("conform_retain_all"),
                randomize=self.get_setting("conform_randomize"),
                method=self.get_setting("conform_method"),
                atol=self.get_setting("atol"),
                rtol=self.get_setting("rtol"),
            )
        # energy_delta is not correctly updated at the moment
        return energy_delta

    def empty_inboxes(self):
        for factor_name, factor_node in self.factor_nodes.items():
            factor_node.empty_inbox()
        for var_name, var_node in self.var_nodes.items():
            var_node.empty_inbox()

    def send_messages_all(
            self):
        """
        Send the messages to the nodes synchronously.
        This schedule is robust for loopy
        graphs.
        Note that tt is not optimal for tree graphs.
        """
        for factor_name, factor_node in self.factor_nodes.items():
            # print("computing", factor_name)
            factor_node.send_messages(
                max_rank=self.get_setting("max_rank"),
                damping=self.get_setting("damping"),
                hard_damping=self.get_setting("hard_damping"),
                atol=self.get_setting("atol"),
                rtol=self.get_setting("rtol"),
                retain_all=self.get_setting("belief_retain_all"),
            )
        for var_name, var_node in self.get_var_nodes_d(
                    include_observed=False
                ).items():
            # print("computing", var_name)
            var_node.send_messages(
                damping=self.get_setting("damping"),
                hard_damping=self.get_setting("hard_damping"),
                max_rank=self.get_setting("max_rank"),
                atol=self.get_setting("atol"),
                rtol=self.get_setting("rtol"),
                retain_all=self.get_setting("belief_retain_all"),
            )

    def send_messages_random(
            self):
        """
        Send the messages to the nodes synchronously but with a random subset to control update size.

        At each timestep we randomly select one var node for each factor to send its message.

        We want to relinearize often for this, and we want to use little or no damping
        """
        raise NotImplementedError("not implemented")
        for factor_name, factor_node in self.factor_nodes.items():
            # print("computing", factor_name)
            factor_node.send_messages(
                max_rank=self.get_setting("max_rank"),
                damping=self.get_setting("damping"),
                hard_damping=self.get_setting("hard_damping"),
                atol=self.get_setting("atol"),
                rtol=self.get_setting("rtol"),
                retain_all=self.get_setting("belief_retain_all"),
            )
        for var_name, var_node in self.get_var_nodes_d(
                    include_observed=False
                ).items():
            # print("computing", var_name)
            var_node.send_messages(
                damping=self.get_setting("damping"),
                hard_damping=self.get_setting("hard_damping"),
                max_rank=self.get_setting("max_rank"),
                atol=self.get_setting("atol"),
                rtol=self.get_setting("rtol"),
                retain_all=self.get_setting("belief_retain_all"),
            )


def var_dims_d(var_nodes):
    """
    return the dimensions of each implicated var node
    """
    var_dims = {}
    for k, v in var_nodes.items():
        var_dims[k] = v.get_dim()
    return var_dims


def var_slices_d(var_nodes):
    """
    return slices for each var node
    """
    var_slices = {}
    offset = 0
    for k, v in var_nodes.items():
        next_offset = offset + v.get_dim()
        var_slices[k] = slice(offset, next_offset)
        offset = next_offset
    return var_slices


def slice_by_var_d(var_nodes, mat):
    """
    Utility which returns a dict of views of a large matrix with slices
    corresponding to the var nodes.
    This should turn a matrix into a dict of matrices.
    """
    slices = var_slices_d(var_nodes)
    return {
        k: mat[slices[k], :] for k in slices
    }
