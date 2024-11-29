import torch
import gc

def kl_normal(mu1, mu2, var1, var2, eps=1e-4):
    assert mu1.shape == mu2.shape
    assert var1.shape == var2.shape
    size = mu1.shape[0]
    var1[range(size), range(size)] += eps # + epsilon*I to avoid numerical precision problems
    var2[range(size), range(size)] += eps
    return (
        torch.trace(torch.linalg.solve(var2, var1))
        + (mu2 - mu1).adjoint() @ torch.linalg.solve(var2, mu2 - mu1)
        - mu1.shape[0]
        + torch.logdet(var2)
        - torch.logdet(var1)
    )/2

def kl_normal_flipped(mu1,mu2,var1,var2):
    return kl_normal(mu2,mu1,var2,var1)

def wasserstein_normal(mu1, mu2, var1, var2):
    assert mu1.shape == mu2.shape
    assert var1.shape == var2.shape

    def sqrtm(matrix):
        # super slow implementation of the covariance matrix square root
        d, P = torch.linalg.eigh(matrix)
        d = torch.real(d) # remove numerical precision problems
        d = torch.where(d > 0, d, torch.tensor([0]).float().to(DEVICE)) # remove numerical precision
        res = (P * d**(1/2)) @ P.adjoint()
        del d, P
        gc.collect()
        torch.cuda.empty_cache()
        return res

    var1_sqrt = sqrtm(var1)
    new_mat = sqrtm(var1_sqrt @ var2 @ var1_sqrt)

    res = torch.square(torch.norm(mu1 - mu2, p=2)) + torch.trace(var1) + torch.trace(var2) - 2*torch.trace(new_mat)
    del var1_sqrt, new_mat
    gc.collect()
    torch.cuda.empty_cache()
    return res


def bias(mu1, mu2, var1=None, var2=None):
    # simple, NORMALIZED bias
    assert mu1.shape == mu2.shape
    del var1, var2
    return torch.norm(mu1-mu2) / mu1.shape[0]

