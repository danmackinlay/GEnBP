import warnings
import torch

from collections import namedtuple
from time import perf_counter
import numpy as np

from ..hermitian_matrices import DiagonalPlusLowRank
from ..torch_formatting import ensemble_packery
from ..gaussian_statistics import mean_dev_from_ens
from ..utils import isscalar

# Utility struct for holding samplewise updates plus some other cruft
ConditionalUpdate = namedtuple(
    'ConditionalUpdate',
    ['ens', 'sigma2_hat', 'info']
)

def pathwise_condition(
        ens_d,   # Whole factor ensemble as dict of batched vectors
        obs,  # conditioning obs, as dict of individual vectors
        sigma2s={},  # diagonal slack, which may be a per-var dict
        sigma2=0.0,  # default diagonal slack
        atol=0.0,
        rtol=None,
        verbose=0):
    """
    Takes a batched ensemble, conditions on some subset of variates.
    Uses a mix of the new HermitianMatrix and old style stuff
    because of API changes.
    """
    la_time = 0.0
    start_time = perf_counter()
    device = next(iter(ens_d.values())).device
    # order-preserving list difference
    unobserved_sites = list(ens_d.keys())
    for k in obs.keys():
        unobserved_sites.remove(k)
    observed_sites = list(obs.keys())
    observed = [ens_d[k] for k in observed_sites]
    unobserved = [ens_d[k] for k in unobserved_sites]
    try:
        observed_pack, _ = ensemble_packery(observed)
        unobserved_pack, unobserved_unpack = ensemble_packery(unobserved)
    except ValueError as e:
        print("observed", observed)
        print("unobserved", unobserved)
        raise e
    # TODO: the below code predates ConstDiagonalMatrix & could be simplified
    #
    # need to rename that sigma2s to sigma2_default because we upcast it
    sigma2_default = torch.as_tensor(sigma2, device=device)
    # clone sigmas2 for safe mutation
    sigma2s = sigma2s.copy()
    for k in observed_sites:
        sigma2s.setdefault(k, sigma2_default)
    for k, v in sigma2s.items():
        if k in observed_sites:
            sigma2s[k] = torch.as_tensor(v, device=device)
        # else:
        #     # silently ignore
        #     warnings.warn(f"sigma2s[{k}] is not observed")
    # OK, now we need to create the diagonal inflation term over U
    sigma2vec = []
    for k in observed_sites:
        sigma2_k = torch.as_tensor(sigma2s[k], device=device)
        # is it scalar?
        if isscalar(sigma2_k):
            sigma2_k = sigma2_k.expand(ens_d[k].shape[1])
        sigma2vec.append(sigma2_k)
    sigma2 = torch.cat(sigma2vec)

    # concatenate the unobserved sites into a single batch of vectors
    Z = observed_pack(observed)
    U = unobserved_pack(unobserved)
    obs_t = torch.cat([obs[k] for k in observed_sites])
    # shouldn't this be `moments_from_ens`?
    m_z, Z_dev = mean_dev_from_ens(Z)
    m_u, U_dev  = mean_dev_from_ens(U)
    Z_var = DiagonalPlusLowRank.from_factors(
        nugget_t=sigma2, lr_t=Z_dev.T, sgn=1.0)
    # transpose because the batch axis is the first dim in this implementation
    # print("Z_dev.shape", Z_dev.shape)
    pred_err = (obs_t-Z)
    Up = (
        Z_var.solve(pred_err.adjoint(), atol=atol, rtol=rtol)
    )
    # Up = DiagonalPlusLowRank.from_factors(
    #         sigma2,
    #         Z_dev.adjoint()
    #     ).solve(pred_err.adjoint(), atol=atol, rtol=rtol)
    step = U_dev.adjoint() @ (Z_dev @ Up)
    # re-transpose so batch index is first again
    U = U + step.adjoint()
    ens_d = dict([
        (k, v) for k, v in
        zip(unobserved_sites, unobserved_unpack(U))])
    # this looks like it might be a valid guesstimate
    # However currently meaningless because it does not incorporate per-node
    # sigma2
    sigma2_hat = pred_err.var()
    total_time = perf_counter() - start_time
    return ConditionalUpdate(
        ens_d,
        sigma2_hat,
        dict(
            total_time=total_time,
            la_time=la_time,
        )
    )
