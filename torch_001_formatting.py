# %%
import torch
from src.torch_formatting import TensorPackery

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

# %%
from torch import nn
from src.torch_formatting import normalize_model_dtypes

class MixedDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.real_weight = nn.Parameter(torch.randn(10, 10, dtype=torch.float32))
        self.complex_weight = nn.Parameter(torch.randn(10, 10, dtype=torch.cfloat))
        self.real_buffer = torch.randn(10, 10, dtype=torch.float64)
        self.complex_buffer = torch.randn(10, 10, dtype=torch.cdouble)

    def forward(self, x):
        return x @ self.real_weight + x @ self.complex_weight.real


# Initialize the model
model = MixedDtypeModel()

# Before normalization
print("Before normalization:")
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.dtype}")

# Normalize dtypes to torch.float64
normalize_model_dtypes(model, torch.float64)

# After normalization
print("\nAfter normalization:")
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.dtype}")

# %%
