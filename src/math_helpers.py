import torch
from torch.autograd import grad


def jacobian_factory(func):
    def jacobian_func(input_tensor, *func_args, **func_kwargs):
        input_tensor.requires_grad_(True)

        # Call the original function with any additional arguments
        output_tensor = func(input_tensor, *func_args, **func_kwargs)

        # Initialize the Jacobian matrix
        jacobian = torch.zeros(output_tensor.numel(), input_tensor.numel())

        # Compute the Jacobian entries
        for i in range(output_tensor.numel()):
            grad_output = torch.zeros_like(output_tensor)
            grad_output.view(-1)[i] = 1.0
            grad_input = grad(output_tensor, input_tensor, grad_outputs=grad_output, retain_graph=True)[0]
            jacobian[i, :] = grad_input.view(-1)

        return jacobian

    return jacobian_func


def atol_rtol(dtype, m, n=None, atol=0.0, rtol=None):
    if rtol is not None:
        return atol, rtol
    elif rtol is None and atol > 0.0:
        return atol, 0.0
    else:
        if n is None:
            n = m
        # choose bigger of m, n
        mn = max(m, n)
        # choose based on eps for float type
        eps = torch.finfo(dtype).eps
        return 0.0, eps * mn


def convert_2d_to_1d(array_2d):
    """
    Converts a 2D array or a batch of 2D arrays to a 1D array or a batch of 1D arrays.
    Preserves the batch dimension if present.
    """
    return array_2d.view(*array_2d.shape[:-2], -1)


def convert_1d_to_2d(array_1d):
    """
    Converts a 1D array or a batch of 1D arrays to a 2D square array or a batch of 2D square arrays.
    Preserves the batch dimension if present.
    Automatically deduces the grid size from the size of the 1D array.
    Assumes the length of the 1D array is a perfect square.
    """
    grid_size = int(array_1d.shape[-1] ** 0.5)
    return array_1d.view(*array_1d.shape[:-1], grid_size, grid_size)
