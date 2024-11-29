# %%
import torch
import numpy as np
from matplotlib import pyplot as plt
from src.gaussian_bp import GBPSettings, FactorGraph, MeasModel, SquaredLoss

# %%
"""#1D Line Fitting Example"""

#@title Create Custom factors

def height_meas_fn(x: torch.Tensor, gamma: torch.Tensor):
    J = torch.tensor([[1-gamma, gamma]])
    return J @ x

def height_jac_fn(x: torch.Tensor, gamma: torch.Tensor):
    return torch.tensor([[1-gamma, gamma]])

class HeightMeasurementModel(MeasModel):
    def __init__(self, loss: SquaredLoss, gamma: torch.Tensor) -> None:
        MeasModel.__init__(self, height_meas_fn, height_jac_fn, loss, gamma)
        self.linear = True

def smooth_meas_fn(x: torch.Tensor):
    return torch.tensor([x[1] - x[0]])

def smooth_jac_fn(x: torch.Tensor):
    return torch.tensor([[-1., 1.]])

class SmoothingModel(MeasModel):
    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, smooth_meas_fn, smooth_jac_fn, loss)
        self.linear = True

#@title Set parameters
n_varnodes = 20
x_range = 10
n_measurements = 15

gbp_settings = GBPSettings(
    damping = 0.1,
    beta = 0.01,
    num_undamped_iters = 1,
    min_linear_iters = 1,
    dropout = 0.0,
  )

# Gaussian noise measurement model parameters:
prior_cov = torch.tensor([10.])
data_cov = torch.tensor([0.05])
smooth_cov = torch.tensor([0.1])
data_std = torch.sqrt(data_cov)

#@title Create measurements {vertical-output: true}

# Plot measurements
meas_x = torch.rand(n_measurements)*x_range
meas_y = torch.sin(meas_x) + torch.normal(0, torch.full([n_measurements], data_std.item()))
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()

#@title Create factor graph {vertical-output: true}
fg = FactorGraph(gbp_settings)

xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T
for i in range(n_varnodes):
    fg.add_var_node(1, torch.tensor([0.]), prior_cov)

for i in range(n_varnodes-1):
    fg.add_factor(
      [i, i+1],
      torch.tensor([0.]),
      SmoothingModel(SquaredLoss(1, smooth_cov))
      )

for i in range(n_measurements):
    ix2 = np.argmax(xs > meas_x[i])
    ix1 = ix2 - 1
    gamma = (meas_x[i] - xs[ix1]) / (xs[ix2] - xs[ix1])
    fg.add_factor(
      [ix1, ix2],
      meas_y[i],
      HeightMeasurementModel(
          SquaredLoss(1, data_cov),
          gamma
        )
      )

fg.print(brief=True)

#@markdown Beliefs are initialized to zero
# Plot beliefs and measurements
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt='_', color="C0", label='Beliefs')
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()

#@title Solve with GaBP {vertical-output: true}
fg.gbp_solve(n_iters=50)

# Plot beliefs and measurements
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt='_', color="C0", label='Beliefs')
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()

def meas_fn(x):
    length = int(x.shape[0] / 2)
    J = torch.cat((-torch.eye(length), torch.eye(length)), dim=1)
    return J @ x


def jac_fn(x):
    length = int(x.shape[0] / 2)
    return torch.cat((-torch.eye(length), torch.eye(length)), dim=1)


class LinearDisplacementModel(MeasModel):
    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, meas_fn, jac_fn, loss)
        self.linear = True

fg = FactorGraph(gbp_settings)

# Initialize variable nodes for frames with prior
for i in range(size):
    for j in range(size):
        init = torch.FloatTensor([j, i]) + torch.normal(torch.zeros(2), prior_noise_std)
        sigma = prior_sigma
        if i == 0 and j == 0:
            init = torch.FloatTensor([j, i])
            sigma = torch.tensor([0.001, 0.001])
        print(init, sigma)
        fg.add_var_node(2, init, sigma)

for i in range(size):
    for j in range(size):
        if j < size - 1:
            fg.add_factor(
                [i*size + j, i*size + j + 1],
                torch.tensor([1., 0.]) + torch.normal(torch.zeros(2), torch.sqrt(noise_cov[0])),
                CubedDisplacementModel(SquaredLoss(dim, noise_cov))
            )
        if i < size - 1:
            fg.add_factor(
                [i*size + j, (i+1)*size + j],
                torch.tensor([0., 1.]) + torch.normal(torch.zeros(2), torch.sqrt(noise_cov[0])),
                CubedDisplacementModel(SquaredLoss(dim, noise_cov))
            )

# %%
# Unlabled example

fg = gbp.FactorGraph(gbp_settings)

anchor_prior_diag_cov = torch.tensor([1.])
prior_diag_cov = torch.tensor([10.])
fg.add_var_node(1, torch.tensor([0.]), anchor_prior_diag_cov)
fg.add_var_node(1, torch.tensor([1.2]), prior_diag_cov)
fg.add_var_node(1, torch.tensor([3.2]), prior_diag_cov)


meas_dofs = 1
noise_diag_cov = torch.tensor([1.])
transition_stds = 0.5

# fg.add_factor([0, 1], torch.tensor([1.]), LinearDisplacementModel(HuberLoss(meas_dofs, noise_diag_cov, transition_stds)))
# fg.add_factor([1, 2], torch.tensor([1.]), LinearDisplacementModel(HuberLoss(meas_dofs, noise_diag_cov, transition_stds)))
fg.add_factor([0, 1], torch.tensor([1.]), LinearDisplacementModel(SquaredLoss(meas_dofs, noise_diag_cov)))
fg.add_factor([1, 2], torch.tensor([1.]), LinearDisplacementModel(SquaredLoss(meas_dofs, noise_diag_cov)))

fg.print()

print("Initial belief means", fg.belief_means().numpy())

joint = fg.get_joint()
print("MAP: ", fg.MAP().numpy())
# %%
