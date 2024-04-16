"""
Implementation of ensemble calcs.

In the paper the ensembles are represented as matrices of column vectors.
This is transposed with respect to the torch convention of using the first axis
as the batch.

Workaround: inside functions we transpose matrices to match the paper notation,
and transpose back before returning.

Messages, however, have no natural batch dimensions, so we use them as-is.
"""
import torch

from math import sqrt
from .hermitian_matrices import DiagonalPlusLowRank, LowRankHermitian, HermitianMatrix
from .math_helpers import atol_rtol
import warnings
from torch.linalg import matrix_norm
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal


def moments_from_ens(ens, sigma2=0.0):
    """
    Moments-form root-factored ensemble representation from ensemble.
    The mean and variance are the mean and variance of the ensemble.

    Naughtily we are not type-stable, returning different subclasses depending
    upon sigma2.
    """
    m, dev = mean_dev_from_ens(ens)
    if torch.any(torch.as_tensor(sigma2) > 0.0):
        var = DiagonalPlusLowRank.from_factors(
            nugget_t=sigma2, lr_t=dev.adjoint())
    else:
        var = LowRankHermitian(dev.adjoint())
    return m, var


def mean_dev_from_ens(ens):
    """
    Mean and deviation matrix of an ensemble.

    NB: normalizes by ensemble size 1/(n-1), unlike in the paper.

    That was a terrible idea.
    """
    m = ens.mean(dim=0)
    dev = (ens - m) / sqrt(ens.shape[0]-1)
    return m, dev


def ens_from_mean_dev(m, dev):
    """
    Add breve matrix back to mean to recover ensemble.
    This is the inverse of mean_dev_from_ens.
    """
    return m + dev * sqrt(dev.shape[0]-1)


def canonical_from_moments(m, var, atol=0.0, rtol=None, retain_all=False):
    """
    Canonical-form low-rank ensemble representation from moments
    """
    prec = var.pinv(atol=atol, rtol=rtol, retain_all=retain_all)
    return (prec @ m.reshape(-1, 1)).reshape(-1), prec


def moments_from_canonical(e, prec, atol=0.0, rtol=None, retain_all=False):
    """
    Moments-form low-rank ensemble representation from canonical.
    This is exactly the same as canonical_from_moments,
    but there is a separate method for clarity.
    """
    var = prec.pinv(atol=atol, rtol=rtol, retain_all=retain_all)
    return (var @ e.reshape(-1, 1)).reshape(-1), var


def canonical_from_ens(ens, sigma2=0.0, atol=0.0, rtol=None, retain_all=False):
    """
    Canonical-form low-rank ensemble representation from ensemble.
    """
    m, var = moments_from_ens(ens, sigma2,)
    e, prec = canonical_from_moments(
        m, var, atol=atol, rtol=rtol, retain_all=retain_all)
    assert len(e.shape) == 1
    assert e.shape[0] == ens.shape[1]
    return e, prec


def mean_dev_from_moments(m, var):
    """
    Nearly trivial, but not quite, because the variance factors are not
    necessarily centred, so their mean must be added to the mean.

    DANGER: this looks like it does what we want, but in fact only works in the
    case that the covariance is has pure root form but full rank.
    """
    lr_factor = var.lr_factor_t()
    offset = lr_factor.mean(1)
    new_m = m + offset
    new_dev = lr_factor.adjoint() - offset
    return new_m, new_dev


def ens_from_moments(m, var):
    """
    Nearly trivial, but not quite, because the variance factors are not
    necessarily centred, so their mean must be added to the mean.

    DANGER: this looks like it does what we want, but in fact only works in the
    case that the covariance is has pure root form but full rank.
    """
    return ens_from_mean_dev(*mean_dev_from_moments(m, var))


def fake_ens_from_moments(m, var, n_ens=1):
    """Simulate some data from something with just about the right statistics.
    This is not the original ensemble, but it is helpful for visualisation of
    joints.
    """
    if isinstance(var, HermitianMatrix):
        var = var.to_tensor()
    return MultivariateNormal(
        loc=m, covariance_matrix=var).sample((n_ens,))


def fake_ens_from_canonical(e, prec, n_ens=1):
    """Simulate some data from something with just about the right statistics.
    This is not the original ensemble, but it is helpful for visualisation of
    joints.
    """
    m, var = moments_from_canonical(e, prec)
    return fake_ens_from_moments(m, var, n_ens=n_ens)


def ens_log_prob(
            obs,
            ens,
            tau2,
        ):
    """
    A Gaussian likelihood.
    Potentially expensive.
    Likelihood evaluation is only used to assess model quality;
    it does not count against the method time cost for fixed hyperparameters.

    I suspect this is no longer used
    """
    m, dev = mean_dev_from_ens(ens)
    return LowRankMultivariateNormal(
        m, dev, torch.as_tensor(tau2, device=m.device).tile(m.shape[0])
    ).log_prob(obs)


def residual_from_moments(moments, obs, weight=False, *args, **kwargs):
    """
    residual energy of a moments dist relative to observations.
    """
    m, var = moments
    if weight:
        e, prec = canonical_from_moments(m, var, *args, **kwargs)
        return (m - obs.ravel()) @ prec
    else:
        return m - obs.ravel()


def residual_from_canonical(canonical, obs, weight=False):
    """
    residual energy of a canonical dist relative to observations.
    """
    e, prec = canonical
    m, _ = moments_from_canonical(e, prec)
    if weight:
        return (m - obs.ravel()) @ prec
    else:
        return m - obs.ravel()


def energy_from_moments(moments, obs, weight=False):
    """
    residual energy of a moments dist relative to observations.
    """
    return (residual_from_moments(moments, obs, weight=weight)**2).sum()


def energy_from_canonical(canonical, obs, weight=False):
    """
    residual energy of a canonical dist relative to observations.
    """
    return (residual_from_canonical(canonical, obs, weight=weight)**2).sum()


def conform_ensemble(method="lstsq", *args, DEBUG_MODE=False, **kwargs):
    if method == "lstsq":
        conformer = conform_ensemble_lstsq
    # elif method == "naive":
    #     conformer = conform_ensemble_naive
    else:
        raise ValueError(f"unknown conformer method {method}")
    return conformer(*args, DEBUG_MODE=DEBUG_MODE, **kwargs)


def conform_ensemble_lstsq(
        X_ens, target_mean, target_var,
        eta2=0.0,
        rtol=0.0,
        atol=0.0,
        retain_all=True,
        randomize=True,
        DEBUG_MODE=False,
        **opt_args):
    """
    Conform an ensemble by least squares optimisation.

    rtol and atol are not quite the same as in `pinverse`;
    Depending upon args they may add diagonal noise to the transform
    so that no modes are lost, or simply take some eigenvecs to 0.

    Deviation from notation in the paper:
    Here Xb is normalized and needs
    the (N-1) factor multiplied in to recover the ensemble.
    """
    # print("conforming ensemble", n_iter, weight_decay, base_lr)
    _, Xb_tp = mean_dev_from_ens(X_ens)
    # ensemble deviations are transposed wrt paper notation because of pytorch
    # batching.
    # Xb_tp is short and wide, but Xb in paper is tall and skinny

    # We need to turn this into a plain matrix approx problem, in terms of low
    # rank factors
    # # The below looks weird but negative V_sig is actually fine
    V_sig = target_var.nugget_t() - eta2
    ## DEBUG suppress eta2?
    # V_sig = target_var.nugget_t()
    ## END DEBUG
    # if torch.any(V_sig < 0.):
    #     warnings.warn(f"potential negative variance {V_sig.p}")
    L = target_var.lr_factor_t()

    M, Lam, Q, _residual, _rresidual = opt_M_lstsq(
        Xb_tp.adjoint(), L, V_sig, rtol=rtol, atol=atol)

    if not torch.all(Lam > 0.):
        warnings.warn(
            f"projection is not full rank, "
            f"{(Lam>0.).sum()} of {Lam.numel()} eigenvalues are positive")
    # Now what to do with small eigs?
    atol, rtol = atol_rtol(M.dtype, M.shape[0], atol=atol, rtol=rtol)
    eigen_floor = (rtol * Lam[-1]) + atol

    if retain_all:
        Lam = torch.clamp(
            Lam, min=eigen_floor)
        # # or
        # Lam = torch.clamp(
            # Lam + eigen_floor, min=0.)
    else:
        first_nonzero = torch.searchsorted(
            Lam, torch.as_tensor(eigen_floor, dtype=Lam.dtype))
        # trim null eigenvecs.
        # we keep one "extra" eigenvector, because the centering
        # matrix creates a zero eigenvalue
        first_nonzero = torch.max(
            first_nonzero-1,
            torch.tensor(0, dtype=torch.long) # do we ever get full rank by accident?
        )
        Q = Q[:, first_nonzero:]
        Lam = Lam[first_nonzero:]
    if len(Lam) < 2:
        raise ValueError(
            "Variance undefined for <2 ensemble")
    if len(Lam) < 4:
        warnings.warn("ensemble is small (<4)")
    if len(Lam) < M.shape[0]:
        warnings.warn("lost some rank")

    # we want a square root
    MLam = torch.sqrt(Lam)
    ## DEBUG inefficient diag
    T = Q * MLam
    ## END DEBUG
    if randomize:
        # perturb the transformation by a unitary rotation.
        # construct a unitary rotation by QR decomp of normalized gaussian matrix.
        # Handy to make sure the variance is carried by many samples if we are
        # in a low rank optimum.
        # Also useful for visualisation
        noise = torch.randn(T.shape[1], T.shape[1])
        noise, _ = torch.linalg.qr(noise)
        # normalize to unitary
        noise /= torch.linalg.norm(noise, dim=0)
        if matrix_norm(T @ T.T - T @ noise @ noise.T @ T.T)/ matrix_norm(T @ T.T) > 1e-6:
            raise ValueError("noise not unitary")
        # rotate
        T = T @ noise
    _energy_delta = 0.  # not actually used atm
    XTb_ = (Xb_tp.adjoint() @ T.detach()).adjoint()
    # Is it even centred bro?
    # Not necessarily, because M is underspecified
    XT_m = XTb_.mean(dim=0)
    ## DEBUG centering
    XTb = XTb_ - XT_m
    if (
        DEBUG_MODE and
        not torch.allclose(XTb.T @ XTb, XTb_.T @ XTb_, rtol=1e-2)
    ):
        from .plots import pred_target_heatmap
        from matplotlib import pyplot as plt
        plt.clf()
        plt.figure(figsize=(15, 15))
        pred_target_heatmap(
             XTb.T @ XTb, XTb_.T @ XTb_
        )
        plt.title("target vs centered var")
        plt.show()
    ## END DEBUG
    new_ens = ens_from_mean_dev(
        target_mean, XTb)
    ## DEBUG did we do what we intended?
    if DEBUG_MODE:
        from .plots import pred_target_heatmap
        from matplotlib import pyplot as plt
        actual_new_mean, actual_new_var = moments_from_ens(new_ens, eta2)
        plt.clf()
        plt.plot(actual_new_mean-target_mean)
        plt.title("target vs actual mean")
        plt.show()
        plt.clf()
        target_var_t = target_var.to_tensor()
        actual_var_t = actual_new_var.to_tensor()
        mask = 1.0 - torch.eye(target_var_t.shape[0])
        pred_target_heatmap(
            actual_var_t*mask, target_var_t*mask
        )
        plt.title("target vs actual var")
        plt.show()
        rerror = matrix_norm(
            (actual_var_t - target_var_t)*mask
            )/matrix_norm(actual_var_t*mask)
        # if rerror > 1.:
        #     # suspiciously terrible approximation
        #     raise ValueError(f"var rerror {rerror}")
        print("var rerror", rerror)
        ## END DEBUG

    return new_ens, _residual, _energy_delta


def opt_M_lstsq(
        Xb, L,
        V,  # NB vector or scalar, not matrix
        atol=0.0, rtol=None,):
    """
    What is the best transform to recover the desired target moments?

    Conform the ensemble using quadratic loss.
    We find M=T@T.adjoint() by solving a linear system.

    Two difficulties which turn out to be annoying in concert:

    1. all terms are hermitian
    2. Xb.T @ Xb always has at least 1 null eigenvalue (because centered)

    Deviation from the paper: Xb is normalized and
    does not need the (N-1) factor included.

    Since we usually want to decompose M into T@T.adjoint(), we can save time
    by returning the Q, Lam decomposition of M, which can produce such a
    decomp, and will additionally tell us if we stayed full rank for no added
    cost.

    NB this doesn't exploit the hermitian quality of the target matrix.
    Could update to use pivoted LDL solves.

    CGD is probably tractable here; we could use the KeOps implementation.
    """
    if rtol is None or rtol<1e-7:
        warnings.warn(f"rtol too small {rtol}, do you know what you are doing?")
    atol, rtol = atol_rtol(
        Xb.dtype, Xb.shape[1], atol=atol, rtol=rtol)

    # Construct the low-rank objective
    # this could be compacted
    # but I want to inspect the intermediate terms for debugging
    XLLX = Xb.adjoint() @ L
    XLLX = XLLX @ XLLX.adjoint()
    XVX = (Xb.adjoint() * V) @ Xb
    XXMXX = XLLX - XVX

    XX = Xb.adjoint() @ Xb

    ## TODO: pivoted LDL decomp would be more efficient. And more stable?
    # https://pytorch.org/docs/stable/generated/torch.linalg.ldl_solve.html
    # LD, pivots, info = torch.linalg.ldl_factor_ex(XX)
    # Q = torch.linalg.ldl_solve(LD, pivots, XXMXX)
    # Q = torch.linalg.ldl_solve(LD, pivots, Q.adjoint())
    # However, AFAICT this won't work with the rank-deficient basis

    # ==
    # # lstsq solve-based method
    # # Difficulty: everything is supposed to be hermitian, we know some stuff about the solution which is not enforced by the solver.
    # # specifically it doesn't hermitian-ness
    # # The solution we get here is not Hermitian
    # lstsq1 = torch.linalg.lstsq(XX, XXMXX, rcond=rtol)
    # assert torch.allclose(XX @ lstsq1.solution, XXMXX)
    # # This is *no longer* a solution; my intuitions about lstsq are bad
    # MXX = 0.5*(lstsq1.solution+lstsq1.solution.adjoint())
    # assert torch.allclose(XX @ MXX, XXMXX)
    # lstsq2 = torch.linalg.lstsq(XX, MXX, rcond=rtol)
    # M = 0.5 * (lstsq2.solution + lstsq2.solution.adjoint())
    # assert torch.allclose(M @ XX, MXX)
    # assert torch.allclose(XX @ M @ XX, XXMXX)
    # Lam, Q = torch.linalg.eigh(M)
    # always clamp -ve eigenvalues
    # Lam = torch.maximum(Lam, torch.tensor(0.0))
    # M = (Q * Lam) @ Q.adjoint()
    # residual = ((XXMXX - XX @ M @ XX)**2).sum()
    # relative residual also of interest
    # rresidual = residual / (XXMXX**2).sum()
    # print(f"rresidual {rresidual}")

    # ==
    # spectrum method
    # eigh-based method (could be SVD in Xb; is that faster?)
    LamX, QX = torch.linalg.eigh(XX)
    # biggest eig last
    s1 = LamX[-1]
    zeroish = LamX < s1 * rtol + atol
    LamX = 1. / LamX[~zeroish]
    QX = QX[:, ~zeroish]
    XXi = (QX * LamX) @ QX.adjoint()
    M2 = XXi @ XXMXX @ XXi
    residual = matrix_norm(XX @ M2 @ XX - XXMXX)
    # relative residual
    rresidual = residual/matrix_norm(XXMXX)
    if rresidual > 1.:
        warnings.warn(f"conformation rresidual large {rresidual}")

    Lam2, Q2 = torch.linalg.eigh(M2)
    # always clamp -ve eigenvalues
    Lam2 = torch.maximum(Lam2, torch.tensor(0.0))
    M2 = (Q2 * Lam2) @ Q2.adjoint()

    residual2 = matrix_norm(XX @ M2 @ XX - XXMXX)
    # relative residual also of interest
    rresidual2 = residual2/matrix_norm(XXMXX)
    if rresidual2 > 1.:
        # suspiciously terrible approximation
        warnings.warn(f"conformation error really large after projection {rresidual2}")

    # if rresidual > 1.:
    #     from .plots import pred_target_heatmap
    #     from matplotlib import pyplot as plt
    #     actual_new_mean, actual_new_var = moments_from_ens(new_ens, eta2)
    #     plt.clf()
    #     plt.plot(actual_new_mean-target_mean)
    #     plt.title("target vs actual mean")
    #     plt.show()
    #     plt.clf()
    #     target_var_t = target_var.to_tensor()
    #     actual_var_t = actual_new_var.to_tensor()
    #     mask = 1.0 - torch.eye(target_var_t.shape[0])
    #     pred_target_heatmap(
    #         actual_var_t*mask, target_var_t*mask
    #     )
    #     plt.title("target vs actual var")
    #     plt.show()
    #     rerror = matrix_norm(
    #         (actual_var_t - target_var_t)*mask
    #         )/matrix_norm(actual_var_t*mask)


    return M2, Lam2, Q2, residual2, rresidual2


# def conform_ensemble_projected(
#         X_ens, target_mean, target_var,
#         rtol=0.0,
#         retain_all=True,
#         **opt_args):
#     """
#     Conform the ensemble Russ's projected gradient descent, with the hermitian eigendecomp to handle the M matrix.

#     rtol adds diagonal noise to the transform so that no modes are lost.
#     If it is not full rank then we could instead prune T?
#     """
#     # print("conforming ensemble", n_iter, weight_decay, base_lr)
#     _, Xb_tp = mean_dev_from_ens(X_ens)
#     # ensemble deviations are transposed wrt paper notation because of pytorch
#     # batching.
#     # Xb_tp is short and wide, Xb is tall and skinny
#     Xb = Xb_tp.adjoint()
#     if hasattr(target_var, "lr_factor"):
#         warnings.warn("using low rank loss")
#         _, Lam, Q = opt_M_lr(Xb, target_var)
#     else:
#         warnings.warn("using dense loss")
#         _, Lam, Q = opt_M_dense(Xb, target_var)
#     # We insert diagonal noise here to stop any modes from decaying.
#     # Not sure if the way we do this is wise.
#     if not torch.all(Lam > 0):
#         warnings.warn(
#             f"projection is not full rank, "
#             f"{(Lam>0).sum()} of {Lam.numel()} eigenvalues are positive")
#     if not retain_all:
#         Q = Q[:, Lam > 0]
#         Lam = Lam[Lam > 0]
#     Lam = torch.clamp(torch.sqrt(Lam), min=rtol)
#     # or
#     # Lam += rtol

#     ## DEBUG inefficient diag
#     T = Q @ torch.diagflat(Lam) * (Xb_tp.shape[0] - 1)
#     ## END DEBUG
#     new_ens = target_mean + (Xb_tp.adjoint() @ T.detach()).adjoint()
#     return new_ens, target_var.diag


# def opt_M_projected(
#         Xb, target_var,
#         n_iter=30,
#         base_lr=1.0,
#         min_grad=1e-8,
#         **more_opt_args):
#     """
#     What is the best transform to recover the desired target moments?

#     NB currently missing consistent handling of eta2.

#     Conform the ensemble using projected gradient descent.

#     We find M=T@T.adjoint() by GD with learning rate scaled to problem using
#     low rank loss.

#     There is a loss function `frob2_loss_lowrank_skew` which should do this
#     calc by autograd, but it seemed easier to do it efficiently inline.

#     Since we usually want to decompose M into T@T.adjoint(), we can save time
#     by returning the Q, Lam decomposition of M, which can produce such a
#     decomp, and will additionally tell us if we stayed full rank for no added
#     cost.
#     """
#     print(f"conforming ensemble with {n_iter} iterations")
#     print(f"size {Xb.shape}")
#     n_items = Xb.shape[0]
#     L = target_var.lr_factor
#     V = target_var.diag
#     M = torch.eye(
#         Xb.shape[1],
#         device=Xb.device,
#         requires_grad=False)
#     # optimal lr
#     beta = 2 * ((Xb.adjoint() @ Xb)**2).sum()
#     lr = base_lr/beta
#     XLLX = Xb.adjoint() @ L
#     XLLX = XLLX @ XLLX.adjoint()
#     XX = Xb.adjoint() @ Xb
#     XVX = Xb.adjoint() @ (as_diag_ish(V, Xb.shape[0]) * Xb)
#     for i in range(n_iter):
#         # manual gradient
#         grad = 2*(XX@M@XX/(n_items-1) - XLLX + XVX)/(n_items-1)
#         grad_mag = vector_norm(grad)/grad.numel()
#         M -= lr * grad
#         Lam, Q = torch.linalg.eigh(M)
#         # projection
#         Lam = torch.maximum(Lam, torch.tensor(0.0))
#         M = Q @ (as_diag_ish(Lam, Xb.shape[0]) * Q.adjoint())
#         if grad_mag < min_grad:
#             warnings.warn(
#                 f"converged after {i+1} iterations with grad mag {grad_mag}")
#             break

#     return M, Lam, Q


# def opt_M_dense(
#         Xb, target_var,
#         n_iter=30,
#         base_lr=1.0,  # < 1.0
#         min_grad=1e-8,
#         **more_opt_args):
#     """
#     What is the best transform to recover the desired target moments?

#     Conform the ensemble using projected gradient descent, without exploiting
#     any low-rank structure.

#     We find M=T@T.adjoint() by GD with learning rate scaled to problem using
#     naive loss.

#     Since we usually want to decompose M into T@T.adjoint(), we can save time
#     by returning the (Q, Lam) decomposition of M, which can produce such a
#     decomp, and will additionally tell us if we stayed full rank.
#     """
#     print(f"conforming ensemble with {n_iter} iterations")
#     print(f"size {Xb.shape}")
#     n_items = Xb.shape[0]
#     M = torch.eye(
#         Xb.shape[1],
#         device=Xb.device,
#         requires_grad=True)
#     # optimal lr
#     beta = 2 * ((Xb.adjoint() @ Xb)**2).sum()
#     lr = base_lr/beta
#     for i in range(n_iter):
#         # gradient
#         frob_loss = matrix_norm(
#             Xb @ M @ Xb.adjoint()/(n_items-1)
#             - target_var,
#             "fro"
#         )**2
#         frob_loss.backward()
#         with torch.no_grad():
#             grad_mag = vector_norm(M.grad)/M.grad.numel()
#             M -= lr * M.grad
#             # This *should* be OK, since M should stay hermitian by symmetry of
#             # gradients
#             Lam, Q = torch.linalg.eigh(M)
#             # projection
#             Lam = torch.maximum(Lam, torch.tensor(0.0))
#             M.copy_(
#                 Q @ (as_diag_ish(Lam, Xb.shape[0]) * Q.adjoint()))
#             M.grad.zero_()

#         if grad_mag < min_grad:
#             warnings.warn(
#                 f"converged after {i+1} iterations with grad mag {grad_mag}")
#             break

#     with torch.no_grad():
#         return M, Lam, Q
