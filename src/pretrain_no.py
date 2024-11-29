"""
Pre-training a neural operator to be used as an initialization for a downstream task.
To make this principled, we optimize for a different but related task.

In this case, we set the target parameters different from training parameters but within the same physical model class, i.e., fluid-flow.

HACK warning: we null out the forcing term in the training data, so the model is not trained to predict using the forcing term.
"""

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from neuralop.models import TFNO, FNO
from src import ns_2d, random_fields
import random
import numpy as np

def collect_grad_variances(optimizer, model):
    """
    Collects the gradient variances for each parameter in the model.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        model (nn.Module): The PyTorch model.

    Returns:
        grad_variances (List[torch.Tensor]): List of gradient variance tensors.
    """
    grad_variances = []
    for param in model.parameters():
        if param.grad is None:
            grad_variances.append(torch.zeros_like(param.data))
            continue
        state = optimizer.state[param]
        if "exp_avg" in state and "exp_avg_sq" in state:
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            ## Let's not substract the mean it should be nearly 0
            # var = exp_avg_sq - exp_avg.pow(2)
            var = exp_avg_sq
            grad_variances.append(var.cpu().clone())
        else:
            grad_variances.append(torch.zeros_like(param.data))
    return grad_variances


def train_model(
    typ="FNO",
    n_layers=4,
    hidden_channels=64,
    n_modes=16,
    d=32,
    delta_t=0.01,
    visc=0.1,
    interval=10,
    alpha=2.0,
    tau=8.0,
    num_epochs=5,
    gd_steps_per_epoch=50,
    n_steps=5,
    rank=0.05,
    v_noise_power=1e2,
    seed=42,
):
    # Device configuration
    device = "cpu"

    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # For deterministic behavior (if using CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Directories and file paths
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create a unique parameter string for memoization
    param_string = (
        f"{typ}_layers{n_layers}_proj{hidden_channels}_modes{n_modes}_"
        f"d{d}_dt{delta_t}_visc{visc}_int{interval}_alpha{alpha}_tau{tau}"
        f"_epochs{num_epochs}_steps{gd_steps_per_epoch}_nsteps{n_steps}_rank{rank}_vnoise{v_noise_power}_seed{seed}"
    )
    model_filename = f"no_ns2d_weights_{param_string}.pth"
    grad_var_filename = f"grad_variance_{param_string}.pth"

    model_path = os.path.join(OUTPUT_DIR, model_filename)
    grad_var_path = os.path.join(OUTPUT_DIR, grad_var_filename)

    # Initialize the model architecture
    model_args = {
        "n_modes": (n_modes, n_modes),
        "in_channels": 1,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
    }
    if typ == "FNO":
        model = FNO(**model_args).to(device)
    elif typ == "TFNO":
        model = TFNO(
            factorization="tucker",
            rank=rank,
            **model_args,
        ).to(device)
    else:
        raise ValueError(f"Invalid model type: {typ}")

    # Check if the model checkpoint and gradient variance data exist
    if os.path.exists(model_path) and os.path.exists(grad_var_path):
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        print(f"Model weights loaded from {model_path}")

        # Load gradient variance data
        grad_variances = torch.load(grad_var_path)
        print(f"Gradient variance data loaded from {grad_var_path}")

        return model, grad_variances
    else:
        print("No existing model checkpoint found. Starting training...")

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training parameters
        batch_size = 8
        # Forcing term generator
        grf_f = random_fields.GaussianRF(2, d, alpha=3.5, tau=10)

        # Function to generate initial conditions
        def generate_initial_conditions(batch_size, d):
            grf = random_fields.GaussianRF(2, d, alpha=alpha, tau=tau)
            x0 = grf.sample(batch_size)
            return x0  # Shape: (batch_size, d, d)

        # Function to simulate steps
        def simulate_steps(x0, f, n_steps, delta_t, interval, v_noise_power, visc):
            x = x0
            x_list = []
            y_list = []

            for t in range(n_steps):
                x = x.clone()
                x = ns_2d.navier_stokes_2d_step(
                    x,
                    f=f*0.0,  # Null out the forcing term
                    visc=visc,
                    delta_t=delta_t,
                    interval=interval,
                    v_noise_power=v_noise_power,
                )
                y = x.clone()
                x_list.append(x)
                y_list.append(y)

            x_tensor = torch.stack(x_list, dim=1)
            y_tensor = torch.stack(y_list, dim=1)
            return x_tensor, y_tensor


        # Initial conditions and forcing term
        x0 = generate_initial_conditions(batch_size, d).to(device)
        f = grf_f.sample(batch_size).to(device)

        # Training loop
        for epoch in range(num_epochs):
            for gd_step in range(gd_steps_per_epoch):
                x0 = generate_initial_conditions(batch_size, d).to(device)
                f = grf_f.sample(batch_size).to(device)

                # Simulate time steps and get training data
                x_batch, y_batch = simulate_steps(
                    x0,
                    f,
                    n_steps=n_steps,
                    delta_t=delta_t,
                    interval=interval,
                    v_noise_power=v_noise_power,
                    visc=visc,
                )

                # Flatten the time and batch dimensions
                x_batch = x_batch.reshape(-1, 1, d, d).to(device)
                y_batch = y_batch.reshape(-1, 1, d, d).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                if gd_step % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{gd_step+1}/{gd_steps_per_epoch}], "
                        f"Loss: {loss.item():.6f}"
                    )

        # After training is complete, collect gradient variances
        grad_variances = collect_grad_variances(optimizer, model)

        # Save model and gradient variance data
        torch.save(model.state_dict(), model_path)
        torch.save(grad_variances, grad_var_path)
        print(f"Model weights saved to {model_path}")
        print(f"Gradient variance data saved to {grad_var_path}")

        return model, grad_variances