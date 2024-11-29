"""
Turning a neural operator into a batched function
"""

# %%
#
import torch
import torch.nn as nn
from neuralop.models import TFNO, FNO
from src.torch_formatting import flatten_model_parameters, unflatten_model_parameters_batch, set_model_parameters, flatten_tensor_batch, unflatten_tensor_batch, batched_vector_model
import os

device = 'cpu'

# Intermediate results we do not wish to vesion
LOG_DIR = os.getenv("LOG_DIR", "_logs")
# Outputs we wish to keep
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
# Figures we wish to keep
FIG_DIR = os.getenv("FIG_DIR", "fig")
# Load the model weights

# Initialize the model architecture
# model = FNO(
#     n_modes=(16, 16),
#     in_channels=3,
#     hidden_channels=16,
#     projection_channels=64,
#     factorization="tucker",
#     n_layers=3,
#     rank=0.42,
# )

# model_path = os.path.join(OUTPUT_DIR, "model_weights.pth")
# model.load_state_dict(torch.load(model_path))
# model = model.to(device)
# print(f"Model weights loaded from {model_path}")


# %% Example usage
# Define  model instance
import torch
from neuralop.models import FNO

# Define your model instance
model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=3, out_channels=1)

# Flatten the model parameters and store shapes
params_vector, param_shapes = flatten_model_parameters(model)

# Flatten an example input and store its info
example_input = torch.randn(1, model.in_channels, 64, 64)  # Shape: (1, C_in, H, W)
input_vector_batch, input_info = flatten_tensor_batch(example_input)

# Prepare batches of inputs and parameters
batch_size = 5

# Batch of inputs (can be batch_size or 1)
X_batch = torch.randn(
    batch_size, model.in_channels, 64, 64
)  # Shape: (batch_size, C_in, H, W)
X_vec_batch, _ = flatten_tensor_batch(X_batch)

# Batch of parameters (can be batch_size or 1)
W_vec_batch = params_vector.unsqueeze(0)  # Shape: (1, N_w)
# Apply the batched_vector_model function
y_vec_batch, output_info = batched_vector_model(
    X_vec_batch, W_vec_batch, input_info, param_shapes, model
)

# Unflatten the outputs back to tensor form
Y_batch = unflatten_tensor_batch(y_vec_batch, output_info)

print("broadcast:", Y_batch.shape)  # Should be (batch_size, C_out, H_out, W_out)

# Or use different parameters for each input:
W_vec_batch2 = torch.randn(batch_size, params_vector.numel())  # Shape: (batch_size, N_w)
y_vec_batch2, output_info2 = batched_vector_model(
    X_vec_batch, W_vec_batch2, input_info, param_shapes, model
)

# Unflatten the outputs back to tensor form
Y_batch2 = unflatten_tensor_batch(y_vec_batch2, output_info2)

print("pairwise", Y_batch2.shape)  # Should be (batch_size, C_out, H_out, W_out)


#%%