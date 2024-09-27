import torch
import torch.nn as nn
import warnings

class TensorPackery:
    """
    Class that packs some tensors into a single vector and unpacks them again,
    based on exemplary tensors.

    # Example usage
    tensors = [torch.randn(2, 3), torch.randn(4), torch.randn(1, 2, 3)]
    packer = TensorPackery(tensors)

    # 2D array (batched)
    tensors_batched = [torch.randn(5, *t.shape) for t in tensors]
    packed = packer.pack(tensors_batched)
    print("Packed tensor (batched):", packed)
    unpacked_tensors = packer.unpack(packed)
    print("Unpacked tensors (batched):", unpacked_tensors)

    # 1D array (no batch)
    packed_no_batch = packer.pack(tensors)
    print("Packed tensor (no batch):", packed_no_batch)
    unpacked_tensors_no_batch = packer.unpack(packed_no_batch)
    print("Unpacked tensors (no batch):", unpacked_tensors_no_batch)

    # Introspection
    print("Shapes:", packer.get_shapes())
    print("Sizes:", packer.get_sizes())
    """
    def __init__(self, tensor_iter):
        self.shapes = []
        self.sizes = []

        for t in tensor_iter:
            shape = list(t.shape)  # No batch dimension in the prototype
            size = t.numel()
            self.shapes.append(shape)
            self.sizes.append(size)

    def pack(self, tensors):
        """
        Pack this list of tensors into a single vector.
        """
        assert len(tensors) == len(self.shapes), "Input list must have the same length as the original list of tensors"

        has_batch_dim = tensors[0].dim() == len(self.shapes[0]) + 1

        if not has_batch_dim:
            return torch.cat([t.reshape(-1) for t in tensors])
        else:
            batch_size = tensors[0].shape[0]
            return torch.cat([t.reshape(batch_size, -1) for t in tensors], dim=1)

    def unpack(self, vector):
        """
        Unpack this single vector into a list of tensors.
        """
        has_batch_dim = vector.dim() == 2
        batch_size = vector.shape[0] if has_batch_dim else 1

        total_size = sum(self.sizes)
        assert vector.numel()  == total_size * batch_size, \
            "Input vector must have the correct number of elements."

        repacked_tensors = []
        index = 0
        for size, shape in zip(self.sizes, self.shapes):
            target_shape = [batch_size, *shape] if has_batch_dim else shape

            repacked_tensors.append(vector[..., index:index + size].reshape(*target_shape))
            index += size

        return repacked_tensors

    def get_shapes(self):
        return self.shapes

    def get_sizes(self):
        return self.sizes



def ensemble_packery(example_batches):
    """
    Function that takes an example iterable of batches of vectors and returns two new functions:

    - concatenate_batches: Returns a similar iterable of tensors concatenated along the second dimension.
    - split_into_batches: Takes a concatenated tensor and breaks it up into a list of batches of vectors that match the original inputs.

    The function does not care if the size of the first (batch) dimension changes, but the second dimension (vector lengths) should not change between definition and invocation.

    Example usage:
    example_batches = [torch.randn(5, 2, 3), torch.randn(5, 4), torch.randn(5, 1, 2, 3)]
    concatenate_batches, split_into_batches = ensemble_packery(example_batches)

    concatenated_tensor = concatenate_batches(example_batches)
    print("Concatenated tensor:", concatenated_tensor)

    split_tensors = split_into_batches(concatenated_tensor)
    print("Split tensors:", split_tensors)
    """
    warnings.warn("I'm pretty sure we don't need `ensemble_packery`")
    try:
        vec_lengths = [b.shape[1] for b in example_batches]
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Input list must contain batches of vectors; got {e}")

    def concatenate_batches(batches):
        assert len(batches) == len(example_batches), "Input list must have the same length as the example list of batches"
        return torch.cat(batches, dim=1)

    def split_into_batches(concatenated_tensor):
        assert concatenated_tensor.shape[1] == sum(vec_lengths), "Input tensor must have the same total size in the second dimension as the example list of batches"
        split_tensors = []
        index = 0
        for length in vec_lengths:
            split_tensors.append(concatenated_tensor[:, index:index + length])
            index += length
        return split_tensors

    return concatenate_batches, split_into_batches


import numpy as np
import torch

def neg_log_dens(x_np):
    """
    Computes the negative log density using the PyTorch function `f`.

    Args:
        x_np (np.ndarray): Input position array in NumPy format.

    Returns:
        float: The computed loss value as a scalar.
    """
    # Convert NumPy array to PyTorch tensor
    x_torch = torch.from_numpy(x_np).float()
    # Ensure the tensor does not require gradients for neg_log_dens
    x_torch.requires_grad_(False)
    # Compute the loss using the PyTorch function `f`
    loss = f(x_torch)
    # Extract the scalar value from the tensor
    loss_value = loss.item()
    return loss_value

def grad_neg_log_dens(x_np):
    """
    Computes the gradient of the negative log density using `f`.

    Instantiate the System class with the defined functions
    >>> system = System(neg_log_dens, grad_neg_log_dens=grad_neg_log_dens)

    Args:
        x_np (np.ndarray): Input position array in NumPy format.

    Returns:
        tuple: A 2-tuple where the first element is the gradient array (np.ndarray)
               and the second element is the loss value (float).
    """
    # Convert NumPy array to PyTorch tensor and enable gradient computation
    x_torch = torch.from_numpy(x_np).float().requires_grad_(True)
    # Compute the loss using the PyTorch function `f`
    loss = f(x_torch)
    # Extract the scalar value from the tensor
    loss_value = loss.item()
    # Compute gradients
    loss.backward()
    # Convert gradients back to NumPy array
    grad_np = x_torch.grad.detach().numpy()
    return grad_np, loss_value
