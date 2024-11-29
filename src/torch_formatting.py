"""
Various versions of tensor packing and unpacking functions.
We experimented with different APIs and desperately need to consolidate.
"""
import warnings
from collections import namedtuple

import torch
import torch.nn as nn


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

ParamInfo = namedtuple("ParamInfo", ["shape", "num_elements", "is_complex", "dtype"])
TensorInfo = namedtuple("TensorInfo", ["shape", "is_complex", "dtype"])


def flatten_model_parameters(model):
    """
    Flattens the parameters of a PyTorch model into a single vector and stores their shapes and types.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        W_vector (torch.Tensor): Flattened parameter vector of shape (N_w,).
        W_shapes (List[ParamInfo]): List containing parameter info.
    """
    W_shapes = []
    W_vector = []
    for param in model.parameters():
        shape = param.shape
        num_elements = param.numel()
        is_complex = torch.is_complex(param)
        dtype = param.dtype
        W_shapes.append(
            ParamInfo(
                shape=shape,
                num_elements=num_elements,
                is_complex=is_complex,
                dtype=dtype,
            )
        )
        if is_complex:
            real_part = param.detach().real.view(-1)
            imag_part = param.detach().imag.view(-1)
            W_vector.append(torch.cat([real_part, imag_part]))
        else:
            W_vector.append(param.detach().view(-1))
    W_vector = torch.cat(W_vector)
    return W_vector, W_shapes


def unflatten_model_parameters_batch(W_vector_batch, W_shapes):
    """
    Unflattens a batch of parameter vectors back into lists of parameter tensors using stored shapes.

    Args:
        W_vector_batch (torch.Tensor): Batch of flattened parameter vectors of shape (batch_size, N_w).
        W_shapes (List[ParamInfo]): List containing parameter info.

    Returns:
        W_tensors_batch (List[List[torch.Tensor]]): List of parameter tensors per sample.

    Raises:
        ValueError: If the vector size does not match the expected total size.
    """
    batch_size = W_vector_batch.shape[0]
    total_expected_size = sum(
        info.num_elements * (2 if info.is_complex else 1) for info in W_shapes
    )
    if W_vector_batch.shape[1] != total_expected_size:
        raise ValueError(
            f"Parameter vector size {W_vector_batch.shape[1]} does not match expected total size {total_expected_size}."
        )

    W_tensors_batch = []
    for b in range(batch_size):
        W_vector = W_vector_batch[b]
        W_tensors = []
        idx = 0
        for info in W_shapes:
            num_elements = info.num_elements
            shape = info.shape
            is_complex = info.is_complex
            dtype = info.dtype
            if is_complex:
                # Read real and imaginary parts
                real_part = W_vector[idx : idx + num_elements].view(shape)
                idx += num_elements
                imag_part = W_vector[idx : idx + num_elements].view(shape)
                idx += num_elements
                # Reconstruct complex tensor
                param_tensor = torch.complex(real_part, imag_part).to(dtype)
            else:
                param_tensor = W_vector[idx : idx + num_elements].view(shape).to(dtype)
                idx += num_elements
            W_tensors.append(param_tensor)
        W_tensors_batch.append(W_tensors)
    return W_tensors_batch


def set_model_parameters(model, W_tensors):
    """
    Sets the parameters of a PyTorch model from a list of parameter tensors.

    Args:
        model (nn.Module): The PyTorch model.
        W_tensors (List[torch.Tensor]): List of parameter tensors.
    """
    param_list = list(model.parameters())
    for param, new_param in zip(param_list, W_tensors):
        if param.data.shape != new_param.shape:
            raise ValueError(
                f"Shape mismatch: parameter shape {param.data.shape}, new_param shape {new_param.shape}"
            )
        param.data.copy_(new_param.data)


def flatten_tensor_batch(tensor_batch):
    """
    Flattens a batch of tensors into a batch of vectors and stores their shapes and types.

    Args:
        tensor_batch (torch.Tensor): Tensor of shape (batch_size, *).

    Returns:
        vector_batch (torch.Tensor): Flattened vectors of shape (batch_size, N).
        tensor_info (TensorInfo): Information about the tensor shape, complex flag, and dtype.
    """
    batch_size = tensor_batch.shape[0]
    shape = tensor_batch.shape[1:]  # Exclude batch dimension
    is_complex = torch.is_complex(tensor_batch)
    dtype = tensor_batch.dtype
    if is_complex:
        real_part = tensor_batch.real.view(batch_size, -1)
        imag_part = tensor_batch.imag.view(batch_size, -1)
        vector_batch = torch.cat([real_part, imag_part], dim=1)
    else:
        vector_batch = tensor_batch.view(batch_size, -1)
    tensor_info = TensorInfo(shape=shape, is_complex=is_complex, dtype=dtype)
    return vector_batch, tensor_info


def unflatten_tensor_batch(vector_batch, tensor_info):
    """
    Unflattens a batch of vectors back into tensors of a given shape.

    Args:
        vector_batch (torch.Tensor): Flattened vectors of shape (batch_size, N).
        tensor_info (TensorInfo): Information about the tensor shape, complex flag, and dtype.

    Returns:
        tensor_batch (torch.Tensor): Tensor of shape (batch_size, *).
    """
    shape = tensor_info.shape
    is_complex = tensor_info.is_complex
    dtype = tensor_info.dtype
    expected_size = torch.prod(torch.tensor(shape)).item()
    if is_complex:
        total_expected_size = expected_size * 2
        if vector_batch.shape[1] != total_expected_size:
            raise ValueError(
                f"Each vector size {vector_batch.shape[1]} does not match expected size {total_expected_size} for shape {shape}."
            )
        batch_size = vector_batch.shape[0]
        real_part = vector_batch[:, :expected_size].view((batch_size,) + shape)
        imag_part = vector_batch[:, expected_size:].view((batch_size,) + shape)
        tensor_batch = torch.complex(real_part, imag_part).to(dtype)
    else:
        if vector_batch.shape[1] != expected_size:
            raise ValueError(
                f"Each vector size {vector_batch.shape[1]} does not match expected size {expected_size} for shape {shape}."
            )
        tensor_batch = vector_batch.view((vector_batch.shape[0],) + shape).to(dtype)
    return tensor_batch


def batched_vector_model(X_vec_batch, W_vec_batch, X_info, W_shapes, model):
    """
    Applies the model to batches of inputs and parameters with shape validation.

    Args:
        X_vec_batch (torch.Tensor): Batch of flattened input vectors, shape (batch_size, N_x)
        W_vec_batch (torch.Tensor): Batch of flattened parameter vectors, shape (batch_size, N_w)
        X_info (TensorInfo): Information about the input tensor shape, complex flag, and dtype.
        W_shapes (List[ParamInfo]): List of parameter info.
        model (nn.Module): An instance of the model to use.

    Returns:
        y_vec_batch (torch.Tensor): Batch of flattened output vectors, shape (batch_size, N_y)
        output_info (TensorInfo): Information about the output tensor shape, complex flag, and dtype.

    Raises:
        ValueError: If the vector sizes do not match the expected sizes.
    """
    batch_size_x = X_vec_batch.shape[0]
    batch_size_w = W_vec_batch.shape[0]

    # Determine the batch size to use
    if batch_size_x == batch_size_w:
        batch_size = batch_size_x
    elif batch_size_x == 1 and batch_size_w > 1:
        batch_size = batch_size_w
        X_vec_batch = X_vec_batch.expand(batch_size, -1)
    elif batch_size_w == 1 and batch_size_x > 1:
        batch_size = batch_size_x
        W_vec_batch = W_vec_batch.expand(batch_size, -1)
    else:
        raise ValueError(
            f"Batch sizes of X_vec_batch ({batch_size_x}) and W_vec_batch ({batch_size_w}) are incompatible."
        )

    y_vec_list = []
    output_info = None  # Will be set after the first pass

    # Unflatten all parameter vectors in the batch
    W_tensors_batch = unflatten_model_parameters_batch(W_vec_batch, W_shapes)

    # Reuse the same model instance
    for i in range(batch_size):
        # Set the model parameters for this sample
        W_tensors = W_tensors_batch[i]
        set_model_parameters(model, W_tensors)

        # Unflatten the input for this sample
        x_vector = X_vec_batch[i]
        X_i = unflatten_tensor_batch(
            x_vector.unsqueeze(0), X_info
        )  # Shape: (1, *X_info.shape)

        # Apply the model
        with torch.no_grad():  # Disable gradient computation if not needed
            assert X_i.shape[0] == 1, "Batch size must be 1"
            assert (
                X_i.shape[1:] == X_info.shape
            ), "Input shape does not match expected shape"
            y_i = model(X_i)  # Shape: (1, *output_shape)

        # Flatten the output and collect
        y_i_vector, y_info = flatten_tensor_batch(y_i)
        if output_info is None:
            output_info = y_info
        else:
            if y_info != output_info:
                raise ValueError(
                    f"Output info {y_info} does not match expected info {output_info}."
                )
        y_vec_list.append(y_i_vector.squeeze(0))

    # Stack the outputs
    y_vec_batch = torch.stack(y_vec_list)
    return y_vec_batch, output_info


def get_model_structure(model):
    model_structure = {}

    for module_name, module in model.named_modules():
        if module_name not in model_structure:
            model_structure[module_name] = {}

        for param_name, param in module.named_parameters(recurse=False):
            model_structure[module_name][param_name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "device": str(param.device),
            }

    return model_structure

def flatten_gradient_variances(grad_variances, W_shapes):
    """
    Flattens the gradient variances into a single vector, consistent with the model parameters.

    Args:
        grad_variances (List[torch.Tensor]): List of gradient variance tensors.
        W_shapes (List[ParamInfo]): List containing parameter info from flatten_model_parameters.

    Returns:
        grad_variance_vector (torch.Tensor): Flattened gradient variance vector of shape (N_w,).
    """
    grad_variance_vector = []
    for var, info in zip(grad_variances, W_shapes):
        if info.is_complex:
            real_part = var.real.view(-1)
            imag_part = var.imag.view(-1)
            grad_variance_vector.append(torch.cat([real_part, imag_part]))
        else:
            grad_variance_vector.append(var.view(-1))
    grad_variance_vector = torch.cat(grad_variance_vector)
    return grad_variance_vector


def normalize_model_dtypes(model, target_dtype):
    """
    Recursively normalize all tensor dtypes in a PyTorch module to match the target_dtype.
    This function modifies the model in place.

    Args:
        model (torch.nn.Module): The model to normalize.
        target_dtype (torch.dtype): The target dtype. Can be a float type (e.g., `torch.float64`).

    Notes:
        - If `target_dtype` is a real type (`torch.float32`, `torch.float64`), it maps complex dtypes
          to their corresponding complex type (`torch.cfloat`, `torch.cdouble`).
        - If `target_dtype` is a complex type, it maps real dtypes to their compatible real type.
    """
    if target_dtype not in [torch.float32, torch.float64, torch.cfloat, torch.cdouble]:
        raise ValueError(
            "Target dtype must be one of: torch.float32, torch.float64, torch.cfloat, torch.cdouble."
        )

    # Determine compatible real and complex types
    if target_dtype in [torch.float32, torch.float64]:
        compatible_real_dtype = target_dtype
        compatible_complex_dtype = (
            torch.cfloat if target_dtype == torch.float32 else torch.cdouble
        )
    else:
        compatible_real_dtype = (
            torch.float32 if target_dtype == torch.cfloat else torch.float64
        )
        compatible_complex_dtype = target_dtype

    for name, param in model.named_parameters(recurse=False):
        if param.is_floating_point():
            param.data = param.data.to(compatible_real_dtype)
        elif param.is_complex():
            param.data = param.data.to(compatible_complex_dtype)
        # No action needed for other types

    for name, buffer in model.named_buffers(recurse=False):
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(compatible_real_dtype)
        elif buffer.is_complex():
            buffer.data = buffer.data.to(compatible_complex_dtype)
        # No action needed for other types

    for child in model.children():
        normalize_model_dtypes(child, target_dtype)