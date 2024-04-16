"""
Poor-man’s linear_operator.

Many classes that wrap a hermitian-matrix-like object and provide
a consistent interface.

We do not support batching.
"""

import torch
from torch.linalg import cholesky_ex, inv_ex, eigh
import warnings
from .utils import isscalar
from torch.linalg import matrix_norm, vector_norm
from .math_helpers import atol_rtol


class LinearOperatorError(Exception):
    pass


class NanError(LinearOperatorError):
    pass


class NotPSDError(LinearOperatorError):
    pass



class HermitianMatrix:
    """
    A horrible python class to use python's horrible object model to implement
    some linear algebra.

    Design notes: dunder methods applied to a tensor should return a tensor.
    dunder methods which are applied to another Hermitian matrix and which
    would preserve Mermitianness should return a HermitianMatrix object.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def rank(self):
        return self.size()

    def is_actually_lr(self):
        "Are my factors lower-rank than I am?"
        return False

    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            return self.mul_tensor(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self.rmul_tensor(other)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self.matmul_tensor(other)
        elif isinstance(other, HermitianMatrix):
            return self.matmul_dense(other)
        return NotImplemented

    def __rmatmul__(self, left):
        if isinstance(left, torch.Tensor):
            return self.rmatmul_tensor(left)
        elif isinstance(left, HermitianMatrix):
            return left.matmul_dense(self)
        return NotImplemented

    def __add__(self, right):
        # if isinstance(right, torch.Tensor):
        #     return self.to_tensor() + right
        return NotImplemented

    def __radd__(self, left):
        # return self + left
        return NotImplemented

    def mul(self, scale):
        """
        Multiply by a scalar, but do not cast to tensor.
        """
        return type(self)(self.to_tensor() * scale)

    def matmul_dense(self, other):
        return self.as_dense() @ other.as_dense()

    def rmatmul_dense(self, left):
        return left.as_dense() @ self.as_dense()

    def mul_tensor(self, other):
        return self.to_tensor() * other

    def rmul_tensor(self, left):
        return left * self.to_tensor()

    def matmul_tensor(self, other):
        return self.to_tensor() @ other

    def rmatmul_tensor(self, left):
        return left @ self.to_tensor()

    def as_dense(self):
        """
        Anything can in principle be a DenseHermitianMatrix if we have enough
        memory to realize it.
        """
        return DenseHermitianMatrix(self.to_tensor())

    def solve(self, x, atol=0.0, rtol=None):
        """
        Dangerous call! We use this unstable method per default
        on the assumption the user wants speed over accuracy
        """
        return self.inv(atol=atol, rtol=rtol) @ x

    # def logdet(self):
    #     """
    #     slow.
    #     """
    #     return torch.logdet(self.to_tensor())

    def inv(self, atol=0.0, rtol=None):
        # atol, rtol ignored for full-rank inverse
        warnings.warn("inv is unstable")
        return type(self)(torch.linalg.inv(self.to_tensor()))

    def pinv(self, atol=0.0, rtol=None, **kwargs):
        return type(self)(
            torch.linalg.pinv(
                self.to_tensor(),
                atol=atol, rtol=rtol))

    def frob2_distance(self, other):
        """
        (Frobenius distance)**2 between two matrices.
        """
        return (self.to_tensor() - other.to_tensor()).pow(2).sum()

    def diagnosis(self):
        """
        return a dict of diagnostics
        """
        return dict()

    def reduced_rank(self, rank, atol=0.0, rtol=None):
        """
        No-op for many matrices; subclasses should override.
        """
        return self

    def extract_block(self, start, end):
        return type(self)(self.to_tensor()[start:end, start:end])

    def as_lowrank(self):
        raise ValueError("cannot convert to lowrank")

    def nugget_t(self):
        return torch.tensor(0.0)

    def diag_t(self):
        return torch.diagonal(self.data)

    def diagnostic_subplot(self, axis, normalize=False, **kwargs):
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cont = self.to_tensor()
        if normalize:
            # calc the correlation matrix
            norm = torch.diagflat(torch.diagonal(cont)**(-0.5))
            cont = norm @ cont @ norm
        f = axis.imshow(
            cont.cpu().numpy(),aspect='auto', cmap='viridis')
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(
            f,
            cax=cax,
            location='right',
            anchor=(0, 0.3),
            # shrink=0.
        )
        axis.set_title("dense")
        return f

    def diagnostic_plot(self, title="", **kwargs):
        """
        hacky diagnostic plot
        """
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        fig.suptitle(title)
        self.diagnostic_subplot(fig.gca(), **kwargs)
        return fig


class DenseHermitianMatrix(HermitianMatrix):
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        # # Check if the matrix is Hermitian
        # assert torch.allclose(
        #     self.data, self.data.adjoint()), "Matrix is not Hermitian"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def size(self):
        return self.data.shape[0]

    def dtype(self):
        return self.data.dtype

    def __mul__(self, other):
        if isinstance(other, DenseHermitianMatrix):
            return self.mul_dense(other)
        return super().__mul__(other)

    def __rmul__(self, left):
        if isinstance(left, DenseHermitianMatrix):
            return self.mul_dense(left)
        return super().__rmul__(left)

    def __matmul__(self, other):
        if isinstance(other, DenseHermitianMatrix):
            return other.matmul_dense(self)
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, DenseHermitianMatrix):
            return other.matmul_dense(self)
        else:
            return super().__rmatmul__(other)

    def __add__(self, other):
        if isinstance(other, DenseHermitianMatrix):
            return self.add_dense(other)
        else:
            return super().__add__(other)

    def __radd__(self, left):
        if isinstance(left, DenseHermitianMatrix):
            return left.add_dense(self)
        else:
            return super().__radd__(left)

    def mul_dense(self, other):
        """
        invoke for other DenseHermitianMatrices
        """
        return type(self)(self.to_tensor() * other.to_tensor())

    def matmul_dense(self, other):
        """
        invoke for other DenseHermitianMatrices
        """
        return type(self)(self.to_tensor() @ other.to_tensor())

    def mul_tensor(self, other):
        return self.to_tensor() @ other

    def rmul_tensor(self, left):
        return left @ self.to_tensor()

    def add_dense(self, other):
        return type(self)(self.to_tensor() + other.to_tensor())

    def to_tensor(self):
        # Dense class simply returns data
        # Beware mutation
        return self.data

    def solve(self, x, atol=0.0, rtol=None):
        # atol, rtol ignored for full-rank inverse
        return torch.linalg.solve(self.data, x)

    def inv(self, atol=0.0, rtol=None):
        # atol, rtol ignored for full-rank inverse
        return type(self)(torch.linalg.inv(self.data))

    def pinv(self, atol=0.0, rtol=None):
        return type(self)(
            torch.linalg.pinv(
                self.to_tensor(),
                atol=atol, rtol=rtol))

    def padded(self, before, after):
        """
        pad the matrix with zeros
        """
        return type(self)(
            torch.nn.functional.pad(
                self.data,
                (before, after, before, after),
                "constant", 0))

    def diagnosis(self):
        """
        return a dict of diagnostics
        """
        return dict(
            shape=tuple(self.data.shape),
            magnitude=matrix_norm(self.data, 'fro').item()
        )

    def nugget_t(self):
        return torch.tensor(0.0, device=self.data.device)


class DiagonalMatrix(HermitianMatrix):

    @classmethod
    def from_tensor(cls, tensor, size=None):
        """
        return a ConstDiagonal or Diagonal as needed
        """
        if isscalar(tensor):
            return ConstDiagonalMatrix(tensor, size=size)
        else:
            return cls(tensor)

    def __init__(self, diag):
        self.diag = torch.as_tensor(diag)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.diag})"

    def __mul__(self, other):
        if isinstance(other, DiagonalMatrix):
            return self.mul_diag(other)
        return super().__mul__(other)

    def __rmul__(self, left):
        if isinstance(left, DiagonalMatrix):
            return left.mul_diag(self)
        return super().__rmul__(left)

    def __matmul__(self, right):
        if isinstance(right, DiagonalMatrix):
            return self.matmul_diag(right)
        elif isinstance(right, DenseHermitianMatrix):
            return self.matmul_dense(right)
        else:
            return super().__matmul__(right)

    def __rmatmul__(self, left):
        if isinstance(left, DiagonalMatrix):
            return left.matmul_diag(left)
        elif isinstance(left, DenseHermitianMatrix):
            return self.matmul_dense(left)
        else:
            return super().__rmatmul__(left)

    def __add__(self, right):
        if isinstance(right, DiagonalMatrix):
            return self.add_diag(right)
        elif isinstance(right, DenseHermitianMatrix):
            return self.add_dense(right)
        else:
            return super().__add__(right)

    def __radd__(self, left):
        if isinstance(left, DiagonalMatrix):
            return self.add_diag(left)
        elif isinstance(left, DenseHermitianMatrix):
            return self.add_diag_dense(self, left)
        else:
            return super().__radd__(left)

    def mul(self, scale):
        """
        Multiply by a scalar, but do not cast to tensor.
        """
        return type(self)(self.diag * scale)

    def mul_tensor(self, other):
        return self.diag[:, None] * other

    def rmul_tensor(self, left):
        return left * self.diag

    def matmul_tensor(self, other):
        return self.mul_tensor(other)

    def rmatmul_tensor(self, other):
        return self.rmul_tensor(other)

    def mul_diag(self, other):
        """
        invoke for other DiagonalMatrices
        """
        return type(self)(self.diag * other.diag)

    def matmul_diag(self, other):
        """
        invoke for other DiagonalMatrices
        """
        return self.mul_diag(other)

    def matmul_dense(self, right):
        """
        invoke for DenseHermitianMatrices
        """
        type(right)(self.mul_tensor(right.to_tensor()))

    def rmatmul_dense(self, left):
        """
        invoke for DenseHermitianMatrices
        """
        type(left)(self.rmul_tensor(left.to_tensor()))

    def diagnosis(self):
        return dict(
            shape=tuple(self.diag.shape),
            magnitude=vector_norm(self.diag, 2).item()
        )

    def add_diag(self, other):
        return type(self)(self.diag + other.diag)

    def add_dense(self, dense):
        data = dense.to_tensor()
        d = data.diagonal()
        d += self.diag
        return type(dense)(data)

    def to_tensor(self):
        return torch.diagflat(self.diag)

    def inv(self, atol=0.0, rtol=None):
        # atol, rtol ignored for this inverse
        return type(self)(1.0 / self.diag)

    def pinv(self, atol=0.0, rtol=None, **kwargs):
        atol, rtol = atol_rtol(
            self.dtype(), self.size(), atol=atol, rtol=rtol)
        diag = self.diag.detach().clone()
        s1 = diag.abs().max()
        zeroish = diag.abs() < s1 * rtol + atol
        diag[zeroish] = 0.0
        diag[~zeroish] = 1.0 / diag[~zeroish]
        return type(self)(diag)

    def sqrt(self):
        return type(self)(torch.sqrt(self.diag))

    def full(self, size=None):
        """
        diagonal tensor
        """
        return self

    def full_t(self, size=None):
        return self.diag

    def padded(self, before, after):
        """
        pad the matrix with zeros
        """
        return type(self)(
            torch.nn.functional.pad(
                self.diag, (before, after), "constant", 0))

    def extract_block(self, start, end):
        return type(self)(self.diag[start:end])

    def append(self, other):
        diag = torch.cat([self.diag, other.full_t()])
        return type(self)(diag)

    def nugget_t(self):
        return self.diag

    def diag_t(self):
        return self.nugget_t()

    def size(self):
        return self.diag.shape[0]

    def dtype(self):
        return self.diag.dtype


class ConstDiagonalMatrix(DiagonalMatrix):
    def __init__(self, diag, size=None):
        # size is optional and only needed for some ops
        self.diag = torch.as_tensor(diag)
        self.size = size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.diag}, size={self.size})"

    def __mul__(self, other):
        if isinstance(other, ConstDiagonalMatrix):
            return self.mul_const_diag(other)
        return super().__mul__(other)

    def __rmul__(self, left):
        if isinstance(left, ConstDiagonalMatrix):
            return left.mul_const_diag(self)
        return super().__rmul__(left)

    def __matmul__(self, other):
        if isinstance(other, ConstDiagonalMatrix):
            return self.matmul_const_diag(other)
        else:
            return super().__matmul__(other)

    def __rmatmul__(self, left):
        if isinstance(left, ConstDiagonalMatrix):
            return left.matmul_const_diag(self)
        else:
            return super().__rmatmul__(left)

    def __add__(self, other):
        if isinstance(other, ConstDiagonalMatrix):
            return self.add_const_diag(other)
        else:
            return super().__add__(other)

    def __radd__(self, left):
        if isinstance(left, ConstDiagonalMatrix):
            return left.add_const_diag(self)
        else:
            return super().__radd__(left)

    def mul(self, scale):
        """
        Multiply by a scalar, but do not cast to tensor.
        """
        return type(self)(self.diag * scale, size=self.size)

    def mul_tensor(self, other):
        return self.diag * other

    def rmul_tensor(self, other):
        return other * self.diag

    def matmul_tensor(self, other):
        return self.diag * other

    def rmatmul_tensor(self, left):
        return left * self.diag

    def mul_const_diag(self, other):
        return type(self)(self.diag * other.diag, size=self.size)

    def matmul_const_diag(self, other):
        return type(self)(self.diag * other.diag, size=self.size)

    def add_const_diag(self, other):
        return type(self)(
            self.diag + other.diag, size=self.size)

    def to_tensor(self, size=None):
        if size is None:
            size = self.size or 1
        return torch.diagflat(self.full_t(size))

    def inv(self, atol=0.0, rtol=None):
        # atol, rtol ignored for this inverse
        return type(self)(1.0 / self.diag, size=self.size)

    def pinv(self, atol=0.0, rtol=None, **kwargs):
        # not technically correct if this encodes a zero diagonal
        return self.inv()

    def sqrt(self):
        return type(self)(torch.sqrt(self.diag), size=self.size)

    def full_t(self, size=None):
        """
        diagonal tensor with all elements filled in
        """
        if size is None:
            size = self.size or 1
        return self.diag.repeat(size)

    def full(self, size=None):
        """
        DiagonalMatrix with all elements filled in
        """
        return DiagonalMatrix(self.full_t(size))

    def padded(self, before, after):
        """
        pad the matrix with zeros
        """
        realized = self.full_t(self.size)
        return type(self)(
            torch.nn.functional.pad(
                realized, (before, after), "constant", 0))

    def diagnosis(self):
        info = dict(
            diag=self.diag,
        )
        if self.size is not None:
            info.update(
                magnitude=(vector_norm(self.diag, 2) * self.size).item(),
                size=self.size
            )
        return info

    def extract_block(self, start, end):
        return type(self)(self.diag, size=end-start)

    def append(self, other):
        if isinstance(other, ConstDiagonalMatrix) and self.diag == other.diag:
            size = None
            if self.size is not None and other.size is not None:
                size = self.size + other.size
            return type(self)(
                torch.cat([self.diag]),
                size=size
            )
        else:
            return self.full().append(other)

    def nugget_t(self, size=None):
        return self.full_t(size)


def reduced_factor_rank(lr_factor, rank, atol=0.0, rtol=None):
    """
    SVD-based rank reduction of a low-rank operator.

    TODO: switch to the (?) faster
    https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html
    """
    U, S, _ = torch.linalg.svd(lr_factor, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    atol, rtol = atol_rtol(
        lr_factor.dtype, lr_factor.shape[0], atol=atol, rtol=rtol)
    if atol>0.0 or rtol>0.0:
        s1 = S.abs().max()
        nonzeroish = S.abs() > s1 * rtol + atol
        U = U[:, nonzeroish]
        S = S[nonzeroish]
    return U * S

class LowRankHermitian(HermitianMatrix):
    """
    A = R @ R.T
    """

    @classmethod
    def from_dense(cls, dense, max_rank=99999):
        """
        Given a symmetric positive-definite dense matrix, return a
        LowRankHermitian which is a low-rank approximation to the dense
        matrix.
        """
        new_rank = min(max_rank, dense.shape[0])
        eigs, Q = eigh(dense)
        eigs = eigs[-new_rank:]
        Q = Q[:, -new_rank:]
        return cls(Q * torch.sqrt(eigs))

    # @classmethod
    # def from_dense_inv(cls, dense,
    #         max_rank=99999,
    #         retain_all=False,
    #         atol=0.0, rtol=None):
    #     """
    #     Given a symmetric positive-definite dense matrix, return a
    #     a low-rank approximation to its inverse.

    #     Selecting retain_all=True is pretty weird; it would guarantee that the inverse is still full rank, which is unhelpful
    #     """
    #     atol, rtol = atol_rtol(
    #         dense.dtype,
    #         *dense.shape, atol=atol, rtol=rtol)
    #     dim = dense.shape[0]
    #     new_rank = min(max_rank, dim)
    #     eigs, Q = eigh(dense)
    #     s1 = eigs[-1]
    #     assert s1>0.0, "matrix is negative definite"
    #     zeroish = eigs < s1 * rtol + atol
    #     if retain_all:
    #         warnings.warn("retain_all=True is weird for dense inverse")
    #         # return entire lr factor
    #         eigs[zeroish] = 0.0
    #         reigs = torch.zeros_like(eigs)
    #         reigs[~zeroish] = eigs[~zeroish] ** -0.5
    #     else:
    #         reigs = eigs[~zeroish]
    #         reigs = reigs ** -0.5
    #         Q = Q[:, ~zeroish]
    #     reigs = reigs[-new_rank:]
    #     Q = Q[:, -new_rank:]
    #     return cls(Q * reigs)

    def __init__(
            self,
            lr_factor,):
        self.lr_factor = torch.as_tensor(lr_factor)

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr_factor})"

    def size(self):
        return self.lr_factor.shape[0]

    def dtype(self):
        return self.lr_factor.dtype

    def rank(self):
        return self.lr_factor.shape[1]

    def is_actually_lr(self):
        # are my factors lower-rank than I am?
        return self.rank() < self.size()

    def to_tensor(self):
        return self.lr_factor @ self.lr_factor.adjoint()

    def __mul__(self, other):
        if isinstance(other, DiagonalMatrix):
            return self.mul_diag(other)
        return super().__mul__(other)

    def __rmul__(self, left):
        if isinstance(left, DiagonalMatrix):
            return left.rmul_diag(self)
        return super().__rmul__(left)

    def __matmul__(self, right):
        if isinstance(right, DiagonalMatrix):
            return self.matmul_diag(right)
        elif isinstance(right, DenseHermitianMatrix):
            return self.matmul_dense(right)
        else:
            return super().__matmul__(right)

    def __rmatmul__(self, left):
        if isinstance(left, DiagonalMatrix):
            return self.matmul_diag(left)
        elif isinstance(left, DenseHermitianMatrix):
            return self.rmatmul_dense(left)
        return super().__rmatmul__(left)

    def __add__(self, right):
        if isinstance(right, DiagonalMatrix):
            return self.add_diag(right)
        elif isinstance(right, LowRankHermitian):
            return self.add_lr(right)
        elif isinstance(right, DenseHermitianMatrix):
            return self.add_dense(right)
        else:
            return super().__add__(right)

    def __radd__(self, left):
        if isinstance(left, DiagonalMatrix):
            return self.add_diag(left)
        elif isinstance(left, LowRankHermitian):
            return left.add_lr(self)
        elif isinstance(left, DenseHermitianMatrix):
            return self.radd_dense(left)
        return super().__radd__(left)

    def mul(self, scale):
        """
        Multiply by a scalar, but do not cast to tensor.
        """
        return type(self)(self.lr_factor * torch.as_tensor(scale).sqrt())

    def mul_tensor(self, other):
        """
        if the rank is low, this is efficient.
        """
        return self.lr_factor @ (self.lr_factor.adjoint() * other)

    def rmul_tensor(self, left):
        """
        if the rank is low, this is efficient.
        """
        return (left * self.lr_factor) @ self.lr_factor.adjoint()

    def matmul_tensor(self, other):
        return self.lr_factor @ (self.lr_factor.adjoint() @ other)

    def rmatmul_tensor(self, left):
        return (left @ self.lr_factor) @ self.lr_factor.adjoint()

    def mul_diag(self, right):
        new_factor = self.lr_factor * right.sqrt()
        return type(self)(new_factor)

    def rmul_diag(self, left):
        new_factor = left.sqrt() * self.lr_factor
        return type(self)(new_factor)

    def matmul_diag(self, right):
        new_factor = self.lr_factor @ right.sqrt()
        return type(self)(new_factor)

    def rmatmul_diag(self, left):
        new_factor = left.sqrt() @ self.lr_factor
        return type(self)(new_factor)

    def matmul_dense(self, right):
        return type(right)(self @ right.to_tensor())

    def rmatmul_dense(self, left):
        return type(self)(left.to_tensor() @ self)

    def add_diag(self, other):
        return DiagonalPlusLowRank(other, self)

    def add_lr(self, other):
        new_factor = torch.cat([self.lr_factor, other.lr_factor], dim=1)
        return LowRankHermitian(new_factor)

    def inv(self, *args, **kwargs):
        if self.is_actually_lr():
            warnings.warn(
                "low-rank inverses are not defined, returning pseudo-inverse")
            return self.pinv(*args, **kwargs)
        else:
            return self.inv_dense(*args, **kwargs)

    def inv_dense(self, max_rank=99999, *args, **kwargs):
        """
        Invert the full matrix.
        """
        return type(self).from_dense_inv(
            self.to_tensor(), max_rank=max_rank, *args, **kwargs)

    def pinv(self, atol=0.0, rtol=None, retain_all=False, max_rank=99999):
        """
        We can find the pseudo-inverse by SVD of the lr factors.
        """
        U, S, V = torch.linalg.svd(self.lr_factor, full_matrices=False)

        atol, rtol = atol_rtol(
            self.lr_factor.dtype,
            *self.lr_factor.shape, atol=atol, rtol=rtol)

        s1 = S.abs().max()
        zeroish = S.abs() < s1 * rtol + atol
        if zeroish.all():
            raise ValueError("no non-zero singular values; we are perfectly uninformative")
        if retain_all:
            # return entire lr factor
            S[zeroish] = 0.0
            S[~zeroish] = 1.0 / S[~zeroish]
        else:
            # # No actually, why keep null vectors around?
            S = S[~zeroish]
            U = U[:, ~zeroish]
            S = 1.0 / S

        return type(self)(U * S)

    def reduced_rank(self, rank, atol=0.0, rtol=None, method="svd"):
        """
        SVD-based rank reduction of a low-rank operator.

        TODO: switch to the (?) faster
        https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html
        """
        if method=="svd":
            reduced_factor = reduced_factor_rank(
                self.lr_factor, rank, atol=atol, rtol=rtol)
        else:
            raise ValueError(f"unknown method {method}")
        return type(self)(
            reduced_factor
        )

    def padded(self, before, after):
        """
        pad the matrix with zeros.

        Beware baffling padding arg order for `torch.nn.functional.pad`.
        """
        return type(self)(
            torch.nn.functional.pad(
                self.lr_factor, (0, 0, before, after),
                "constant", 0))

    def solve(self, x, atol=0.0, rtol=None):
        return self.pinv(atol=atol, rtol=rtol) @ x

    def diagnosis(self):
        info = dict(
            lr_shape=tuple(self.lr_factor.shape),
            magnitude=matrix_norm(
                self.lr_factor.adjoint() @ self.lr_factor, 'fro').item()
        )
        return info

    def extract_block(self, start, end):
        return type(self)(self.lr_factor[start:end, :])

    def append(self, other):
        return type(self)(torch.cat([self.lr_factor, other.lr_factor], dim=1))

    def as_lowrank(self):
        return self

    def nugget_t(self):
        return torch.tensor(0.0, device=self.lr_factor.device)

    def diag_t(self):
        return (self.lr_factor**2).sum(dim=1)

    def lr_factor_t(self):
        return self.lr_factor

    def diagnostic_subplot(self, axis, **kwargs):
        """
        class-specific subplot showing lr factor
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        f = axis.imshow(self.lr_factor.cpu().numpy(), aspect='auto', cmap='viridis')
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(
            f,
            cax=cax,
            location='right',
            anchor=(0, 0.3),
            # shrink=0.
        )
        return f

    def diagnostic_plot(self, title="", **kwargs):
        """
        hacky diagnostic plot, showing `lr_factor` if any such exists,
        otherwise the whole thing
        """
        from matplotlib import pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        dense_plot = super().diagnostic_subplot(ax1, **kwargs)
        ax1.set_title("dense")
        lr_plot = self.diagnostic_subplot(ax2, **kwargs)
        ax2.set_title("lr_factor")
        fig.suptitle(f"{title}")
        return fig


def inv_capacitance_sqrt(
        sig2, lr_factor, sgn=1.0,
        max_rank=99999, atol=0.0, rtol=None, retain_all=False):
    """
    sqrt(sig2 + sgn * lr_factor.adjoint() @ lr_factor)**(-1)

    Increases the diagonal until the matrix is invertible and returns the
    nearly-low-rank inverse, with an optional maximum rank.

    While the diagonal inflation is not a weird move for a covariance matrix, I
    know of no meaningful interpretation of it for a precision matrix with
    sgn=-1, so we fret about that.

    TODO: logarithmic speedup if we switch to torch.svd_lowrank.
    """
    atol, rtol = atol_rtol(
        lr_factor.dtype, lr_factor.shape[0], atol=atol, rtol=rtol)
    atol = torch.as_tensor(atol)
    rtol = torch.as_tensor(rtol)
    sig2 = torch.as_tensor(sig2, dtype=lr_factor.dtype)
    kappa2 = torch.reciprocal(sig2)
    if isscalar(kappa2):
        capacitance = kappa2 * lr_factor.adjoint() @ lr_factor
    else:
        capacitance = (
            lr_factor.adjoint()
            @ (kappa2[:, None] * lr_factor))
    d = torch.diagonal(capacitance)
    d += sgn
    Lam, Q = torch.linalg.eigh(sgn * capacitance)

    if not torch.all(Lam > 0.):
        warnings.warn(
            f"matrix has unexpected sign {Lam}")

    # eigs are in ascending order
    if sgn == 1:
        s1 = Lam[-1]
        thresh = s1 * rtol + atol
        if retain_all:
            # inflate diagonal
            smallest = Lam[0]
            if smallest < atol:
                deficit = thresh - smallest
                Lam += deficit
                kappa2 += deficit
                warnings.warn(
                    f"capacitance_inv inflated kappa2={kappa2} by {deficit}")

        else:
            # delete all the small eigs
            keep = Lam > thresh
            if not torch.all(keep):
                warnings.warn(
                    f"capacitance_inv nulled eig={Lam[~keep]}")
            Lam = Lam[keep]
            Q = Q[:, keep]


        if max_rank is not None:
            Lam = Lam[-max_rank:]
            Q = Q[:, -max_rank:]

        # sqrt of inverse
        L = Q * (Lam ** (-0.5))
        return kappa2, L

    elif sgn == -1:
        # trickier!
        # we have negated the matrix to make the capacitance eigs positive.
        # Small eigs are important now
        # All eigs should sandwiched between 0 and |kappa2|;
        # what do we do if they bleed out of *both* sides of that interval?
        # let us be conservative and try not to touch anything
        s1 = Lam[-1]
        thresh = s1 * rtol + atol
        if retain_all:
            # deflate diagonal
            # inflate diagonal
            smallest = Lam[0]
            if smallest < thresh:
                deficit = thresh - smallest
                Lam += deficit
                kappa2 += deficit
                warnings.warn(
                    f"capacitance_inv inflated kappa2={kappa2} by {deficit}")
        else:
            # delete all the small eigs
            keep = Lam > atol
            if any(~keep):
                warnings.warn(
                    f"capacitance_inv nulled eig={Lam[~keep]}")
            Lam = Lam[keep]
            Q = Q[:, keep]

        # actually take sqrt of inverse eigenvals
        invLam = Lam ** (-0.5)

        if max_rank is not None:
            invLam = invLam[-max_rank:]
            Q = Q[:, -max_rank:]

        L = Q * invLam
        return kappa2, L

    else:
        raise ValueError("sgn must be +/-1")


class DiagonalPlusLowRank(HermitianMatrix):
    """
    Container for a diag plus a low-rank matrix times coefficient.

    A = D + sgn * R @ R.T

    Danger: this holds _references_ to the constituent matrices, which might
    get recycled, and thus allow us to mutate the DiagonalPlusLowRank.
    Is this bad?

    This only supports sgn=+/-1; we could be more general.
    That would require complex dtypes, or signed factor blocks.
    """
    @classmethod
    def from_dense(cls, dense, max_rank=99999, nugget_t=None):
        """
        Given a symmetric positive-definite dense matrix, return a
        DiagonalPlusLowRank which is a low-rank approximation to it.
        AFAICT this is natural only for sgn = 1.
        """
        dim = dense.shape[0]
        new_rank = min(max_rank, dim)
        eigs, Q = eigh(dense)
        resid = eigs[:-new_rank].sum()
        eigs = eigs[-new_rank:]
        Q = Q[:, -new_rank:]
        if nugget_t is None:
            nugget_t = resid / dim
            diag = ConstDiagonalMatrix(nugget_t, size=dim)
        else:
            diag = DiagonalMatrix.from_tensor(nugget_t, size=dim)

        return cls(
            diag,
            LowRankHermitian(Q * torch.sqrt(eigs)),
            1.0
        )

    # @classmethod
    # def from_dense_inv(cls,
    #         dense,
    #         nugget_t,
    #         sgn=1.0,  # sign of the low-rank term in the input
    #         max_rank=99999,
    #         retain_all=False,
    #         atol=0.0, rtol=None):
    #     """
    #     Given a symmetric positive-definite dense matrix, return a
    #     DiagonalPlusLowRank which is a low-rank approximation to the inverse of it plus a diagonal.
    #     our target nugget_t is mandatory.
    #     For consistency we want the sign to switch when we calculate
    #     """
    #     dim = dense.shape[0]
    #     new_rank = min(max_rank, dim)
    #     if sgn == -1 and new_rank < dim:
    #         raise ValueError("not sure how to reduce rank in this case")
    #     if sgn == 1:
    #         # D + U @ U.T
    #         sdense = dense * sgn
    #         d = torch.diagonal(sdense)
    #         d -= nugget_t
    #         lr = LowRankHermitian.from_dense_inv(
    #             sdense, max_rank=new_rank, atol=atol, rtol=rtol, retain_all=retain_all)
    #         diag = DiagonalMatrix.from_tensor(1.0/nugget_t, size=dim)
    #     else:
    #         # D - U @ U.T
    #         sdense = dense * sgn
    #         d = torch.diagonal(sdense)
    #         d += nugget_t
    #         lr = LowRankHermitian.from_dense_inv(
    #             sdense, max_rank=new_rank, atol=atol, rtol=rtol, retain_all=retain_all)
    #         diag = DiagonalMatrix.from_tensor(1.0/nugget_t, size=dim)
    #     return cls(
    #         diag,
    #         lr,
    #         -sgn
    #     )

    @classmethod
    def from_factors(cls, nugget_t, lr_t, sgn=1.0):
        """
        Is this naughty?
        We are not type-safe, returning a pure low-rank if null diagonal.
        """
        nugget_t = torch.as_tensor(nugget_t)
        lr_t = torch.as_tensor(lr_t)
        lr = LowRankHermitian(lr_t)
        if torch.allclose(
                nugget_t, torch.tensor(0.0)):
            return lr
        diag = DiagonalMatrix.from_tensor(
            nugget_t, lr.size())
        return cls(
            diag=diag,
            lr_matrix=lr,
            sgn=sgn
        )

    def __init__(self, diag, lr_matrix, sgn=1.0):
        """
        L is a LowRankHermitian
        D is a DiagonalMatrix
        """
        self.diag_matrix = diag
        self.lr_matrix = lr_matrix
        self.sgn = sgn

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(diag={self.diag_matrix}, "
            f"lr={self.lr_matrix}, sgn={self.sgn})"
        )

    def size(self):
        return self.lr_matrix.size()

    def dtype(self):
        return self.lr_matrix.dtype()

    def rank(self):
        """
        abuse of terminology!
        the diagonal term, if non-zero, means that this is full rank, but we
        are happy to ignore that for this use case.
        """
        return self.lr_matrix.rank()

    def is_actually_lr(self):
        return self.rank() < self.size()

    # def __mul__(self, other):
    #     if isinstance(other, LowRankHermitian):
    #         return self.mul_lr(other)
    #     return super().__mul__(other)

    # def __rmul__(self, left):
    #     if isinstance(left, LowRankHermitian):
    #         return left.mul_lr(self)
    #     return super().__rmul__(left)

    def __matmul__(self, other):
        if isinstance(other, DiagonalMatrix):
            return self.matmul_diag(other)
        else:
            return super().__matmul__(other)

    def __rmatmul__(self, left):
        if isinstance(left, DiagonalMatrix):
            return self.matmul_diag(left)
        else:
            return super().__rmatmul__(left)

    def __add__(self, other):
        if isinstance(other, DiagonalMatrix):
            return self.add_diag(other)
        elif isinstance(other, LowRankHermitian):
            return self.add_lr(other)
        elif isinstance(other, DiagonalPlusLowRank):
            return self.add_diaglr(other)
        else:
            return super().__add__(other)

    def __radd__(self, left):
        if isinstance(left, DiagonalMatrix):
            return self.add_diag(left)
        elif isinstance(left, LowRankHermitian):
            return self.add_lr(left)
        elif isinstance(left, DiagonalPlusLowRank):
            return self.add_diaglr(left)
        else:
            return super().__radd__(left)

    def mul(self, scale):
        """
        Multiply by a scalar, but do not cast to tensor.
        This does not support scale <= 0
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        return type(self)(
            self.diag_matrix.mul(scale),
            self.lr_matrix.mul(scale),
            self.sgn
        )

    def mul_tensor(self, other):
        return self.diag_matrix * other + (self.lr_matrix * self.sgn) * other

    def rmul_tensor(self, left):
        return (
            left * self.diag_matrix
            + (left * self.sgn) * self.lr_matrix
        )

    def matmul_tensor(self, other):
        return (
            self.diag_matrix @ other
            + self.lr_matrix @ (self.sgn * other)
        )

    def rmatmul_tensor(self, left):
        return left @ self.diag_matrix + (self.sgn * left) @ self.lr_matrix

    def matmul_diag(self, other):
        return type(self)(
            self.diag_matrix @ other,
            self.lr_matrix @ other,
            self.sgn
        )

    def add_diag(self, other):
        return type(self)(
            self.diag_matrix + other,
            self.lr_matrix,
            self.sgn
        )

    def add_lr(self, other):
        assert self.sgn == 1
        return type(self)(
            self.diag_matrix,
            self.lr_matrix + other.lr_matrix,
            self.sgn
        )

    def add_diaglr(self, other):
        assert self.sgn == other.sgn
        return type(self)(
            self.diag_matrix + other.diag_matrix,
            self.lr_matrix + other.lr_matrix,
            self.sgn
        )

    def to_tensor(self):
        dense = self.sgn * self.lr_matrix.to_tensor()
        diag = torch.diagonal(dense)
        diag += self.diag_matrix.diag
        return dense

    def inv(self, *args, **kwargs):
        if self.is_actually_lr():
            return self.inv_tall(*args, **kwargs)
        else:
            warnings.warn(
                "roots are not low rank; falling back to dense inversion")
            return self.inv_dense(*args, **kwargs)

    def inv_dense(self,
            max_rank=99999, retain_all=False, atol=0., rtol=None,
            *args, **kwargs):
        """
        return a new DiagonalPlusLowRank whose value is
        self^{-1}, by casting to dense and factoring that.
        Cheaper if the factors are wide rather than tall.
        """
        # if max_rank==99999:
        #     raise ValueError("max_rank must be finite")
        warnings.warn(f"dense inverse of {self} with max_rank {max_rank}")
        size = self.size()
        new_rank = min(max_rank, size)
        new_diag = self.diag_matrix.inv()
        if self.sgn == -1:
            PU = torch.linalg.inv(self.to_tensor())
            d = torch.diagonal(PU)
            d -= new_diag.diag
            Lam, Q = torch.linalg.eigh(PU)
        else:
            VK = -torch.linalg.inv(self.to_tensor())
            d = torch.diagonal(VK)
            d += new_diag.diag
            Lam, Q = torch.linalg.eigh(VK)
        atol, rtol = atol_rtol(
            self.dtype(),
            self.size(), atol=atol, rtol=rtol)
        s1 = Lam[-1]
        thresh = s1 * rtol + atol
        keep = Lam > thresh
        # if not torch.all(keep):
        #     warnings.warn(
        #         f"inv_dense nulled eig={Lam[~keep]}")
        if retain_all:
            # return entire lr factor
            Lam[~keep] = 0.0
        else:
            Lam = Lam[keep]
            Q = Q[:, keep]
        Lam = Lam[-new_rank:]
        Q = Q[:, -new_rank:]
        R = Q * (Lam ** 0.5)
        return type(self).from_factors(
            new_diag.diag,
            R,
            self.sgn * -1)

    # def inv_wide(self, max_rank=99999, retain_all=False, *args, **kwargs):
    #     """
    #     return a new DiagonalPlusLowRank whose value is
    #     self^{-1}, by svd of short wide factors.
    #     Actually this is not well-posed if the diagonal is not constant.
    #     """
    #     dense = self.to_tensor()
    #     return type(self).from_dense_inv(
    #         dense, self.diag_matrix.diag, sgn=self.sgn,
    #         max_rank=max_rank, retain_all=retain_all, *args, **kwargs)

    def inv_tall(self, atol=0.0, rtol=None, retain_all=False):
        """
        return a new DiagonalPlusLowRank whose value is
        self^{-1} by Woodbury identities of tall skinny factors.
        """
        lr_factor = self.lr_matrix.lr_factor
        diag = self.diag_matrix.diag
        kappa2 = self.diag_matrix.inv().diag
        sgn = self.sgn
        kappa2_, L = inv_capacitance_sqrt(
            diag, lr_factor, sgn, retain_all=retain_all,
            atol=atol, rtol=rtol)
        if torch.any(kappa2 != kappa2_):
            warnings.warn(
                f"capacitance_inv returned kappa2={kappa2_}!={kappa2}")
            kappa2 = kappa2_

        if isscalar(kappa2):
            R = kappa2 * lr_factor @ L
            return type(self)(
                ConstDiagonalMatrix(kappa2),
                LowRankHermitian(R),
                sgn * -1)
        else:
            R = kappa2[:, None] * lr_factor @ L
            return type(self)(
                DiagonalMatrix(kappa2),
                LowRankHermitian(R),
                sgn * -1)

    def pinv(self, *args, **kwargs):
        """
        return a new DiagonalPlusLowRank whose value is
        self^{-1}
        """
        return self.inv(*args, **kwargs)

    def solve(self, x, atol=0.0, rtol=None):
        """
        For some reason I need to re-implement this on DiagPlusLowRank.
        """
        inv = self.inv(atol=atol, rtol=rtol)
        return inv @ x

    def reduced_rank(self, rank, atol=0.0, rtol=None):
        """
        SVD-based (i.e. Frobenius-error-minising) rank reduction of a low-rank operator.
        Probably not meaningful if sgn=-1,
        Why would we care about Frobenius-approximation of a precision matrix?
        """
        return type(self)(
            self.diag_matrix,
            self.lr_matrix.reduced_rank(rank, atol=0.0, rtol=None),
            self.sgn
        )

    def padded(self, before, after):
        """
        pad the matrix with zeros
        """
        return type(self)(
            self.diag_matrix.full(self.size()).padded(before, after),
            self.lr_matrix.padded(before, after),
            self.sgn
        )

    def diagnosis(self):
        info = dict(
            diag=self.diag_matrix.diagnosis(),
            lr=self.lr_matrix.diagnosis(),
        )
        return info

    def extract_block(self, start, end):
        return type(self)(
            diag=self.diag_matrix.extract_block(start, end),
            lr_matrix=self.lr_matrix.extract_block(start, end),
            sgn=self.sgn)

    def append(self, other):
        return type(self)(
            diag=self.diag_matrix.append(other.diag_matrix),
            lr_matrix=self.lr_matrix.append(other.lr_matrix),
            sgn=self.sgn)

    def as_lowrank(self):
        if self.sgn == -1:
            raise ValueError("cannot convert to lowrank with sgn=-1")
        return self.lr_matrix

    def nugget_t(self):
        return self.diag_matrix.nugget_t()

    def diag_t(self):
        """
        The diagonal of the matrix (not just the diagonal factor)
        """
        return self.diag_matrix.diag_t() + self.lr_matrix.diag_t()

    def lr_factor_t(self):
        return self.lr_matrix.lr_factor

    def diagnostic_plot(self, title="", **kwargs):
        """
        hacky diagnostic plot
        """
        from matplotlib import pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 15))
        dense_plot = super().diagnostic_subplot(ax1, **kwargs)
        ax1.set_title("dense")
        lr_plot = self.lr_matrix.diagnostic_subplot(ax2, **kwargs)
        ax2.set_title("lr_factor")
        diag_plot = self.diag_matrix.diagnostic_subplot(ax3, **kwargs)
        ax3.set_title("diag")
        fig.suptitle(f"{title}")
        return fig


# def lowrank_add(lr_items):
#     """
#     Adds a list of lowrank matrices, and returns a new LowishRankOperator.
#     More efficient than iteratively calling __add__.
#     """
#     type_ = type(lr_items[0])
#     if len(lr_items) > 1 and not all([
#             isinstance(lr, type_) for lr in lr_items[1:]]):
#         raise TypeError(
#             "lowrank_add only knows how to handle homogenous lists " +
#             f"{[type(item) for item in lr_items ]}"
#         )
#     return type_(
#         torch.cat([lr.lr_factor for lr in lr_items], dim=1)
#     )
#
# def lowishrank_add(lr_items):
#     """
#     Adds the list of lowrank matrices, and returns a new LowRankHermitian.
#     """
#     type_ = type(lr_items[0])
#     if len(lr_items) > 1 and not all([
#             isinstance(lr, type_) for lr in lr_items[1:]]):
#         raise TypeError(
#             "lowrank_add only knows how to handle homogenous lists " +
#             f"{[type(item) for item in lr_items ]}"
#         )
#     lr_factor = torch.cat([lr.lr_factor for lr in lr_items], dim=1)
#     if diag_method == "add":
#         diag = sum([lr.diag for lr in lr_items])
#     elif diag_method == "keep":
#         # we are not summing the diagonals, so we want to leave them untouched.
#         # But what if the concatenation destroys the positive or negative definiteness?
#         # Let's see if anything breaks.
#         # This is complicated by the decision to allow diags to be scalars or vectors
#         # do we need that?
#         # first, we look for all-scalars
#         if all([
#                 isscalar(lr.diag)
#                 for lr in lr_items]):
#             # all scalars!
#             diags = torch.cat([lr.diag.reshape(1) for lr in lr_items])
#             diag = torch.max(diags)
#         else:
#             # we have at least one vector
#             diags = torch.stack(
#                 [lr.realize_diag() for lr in lr_items], dim=1)
#             diag = torch.max(diags, dim=1)[0]
#     else:
#         raise ValueError(f"unknown diag_method={diag_method}")
#     return type_(
#         diag=diag,
#         lr_factor=lr_factor
#     )


# def frob2_loss(K1, K2):
#     """
#     For matrices K1,K2,  computes the loss
#     ||K1-K2||_F^2, which is just squared deviation.
#     """
#     return ((K1- K2)**2).sum()


# def frob2_loss_lowrank_naive(delta2, U, R):
#     """
#     For tall skinny matrices U, R, naively computes the loss
#     ||U@U.adjoint() - R@R.adjoint() + delta2.I||_F^2.
#     """
#     K1 = U@U.adjoint() + delta2* torch.eye(U.shape[0])
#     K2 = R@R.adjoint()

#     return frob2_loss(K1, K2)


# def frob2_loss_lowrank(delta2, U, R, UUs=None, XRRX=None, URs=None):
#     """
#     For tall skinny matrices U, R, efficiently computes the loss
#     ||U@U.adjoint() - R@R.adjoint() + delta2.I||_F^2.

#     We allow pre-computed matrices to be passed in, to avoid recomputing them.

#     return the loss, and the delta2 estimate = ((U**2).sum() - (R**2).sum())/D
#     """
#     if U.shape[1] > U.shape[0]:
#         warnings.warn("U is not tall skinny")
#     D = U.shape[0]
#     if UUs is None:
#         UUs = ((U.adjoint() @ U)**2).sum()
#     if XRRX is None:
#         XRRX = ((R.adjoint() @ R)**2).sum()
#     if URs is None:
#         URs = ((U.adjoint() @ R)**2).sum()
#     factor_frob2_delta = ((U**2).sum() - (R**2).sum())
#     # return actual loss estimate, and delta2 estimate
#     return (
#         UUs + XRRX -2 * URs
#         + 2 * delta2 * factor_frob2_delta
#         + delta2**2 * D
#     ), factor_frob2_delta/D
#
#
# def frob2_loss_lowrank_skew(M, Delta, X, R, XX=None, XRRX=None, XDX=None):
#     """
#     For tall skinny matrices X, R, efficiently computes the loss
#     ||X@M@X.adjoint() - R@R.adjoint() + diag(Delta)||_F^2.
#
#     We allow pre-computed matrices to be passed in, to avoid recomputing them.
#
#     return the loss, and the Delta diagonal = ((X**2).sum() - (R**2).sum())/D
#     """
#     if X.shape[1] > X.shape[0]:
#         warnings.warn("X is not tall skinny")
#     D = X.shape[0]
#     if XX is None:
#         XX = ((X.adjoint() @ X)**2)
#     if XRRX is None:
#         XRRX = ((X.adjoint() @ R)**2)
#         XRRX = XRRX * XRRX.adjoint()
#     raise NotImplementedError(
#         "Actually this looks more boring than circumventing autodiff rn")
#     if XDX is None:
#         XDX = ((X.adjoint() @ R)**2)
#     factor_frob2_delta = ((X**2).sum() - (R**2).sum())
#     # return actual loss estimate, and delta estimate
#     return (
#         XX + XRRX -2 * XDX
#         + 2 * Delta * factor_frob2_delta
#         + Delta**2 * D
#     ), factor_frob2_delta/D
