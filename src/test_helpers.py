"""
Helper functions to make contrived test problems.
"""

import torch
import torch.nn.functional as F

##
## 1d testers
##

def make_ball_1d(d, radius, trunc_end=0, dtype=None):
    """returns a 1d torch array representing the discretization of a disc of the given radius which is simply an interval.
    Values are 1 inside the circle and 0 outside
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = torch.linspace(-d, d, d, dtype=dtype)
    disc = ((x**2) < radius**2).type(dtype)
    return torch.roll(disc, trunc_end, dims=(0,))

def convolve_array_1d(a, b):
    """
    Perform a 1D convolution of `a` with `b`.
    `a` can be either 1D or 2D (batch of 1D vectors).
    Both `a` and `b` are assumed to be periodic.
    The output will be the same size as `a`.

    # Example usage
    a_1d = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    a_2d = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32)
    b_odd = torch.tensor([1, 0, -1], dtype=torch.float32)
    b_even = torch.tensor([1, -1], dtype=torch.float32)

    output_1d_odd = periodic_conv1d(a_1d, b_odd)
    output_2d_odd = periodic_conv1d(a_2d, b_odd)
    output_1d_even = periodic_conv1d(a_1d, b_even)
    output_2d_even = periodic_conv1d(a_2d, b_even)

    print("1D Convolution with Odd Kernel:\n", output_1d_odd)
    print("\n2D Convolution with Odd Kernel:\n", output_2d_odd)
    print("\n1D Convolution with Even Kernel:\n", output_1d_even)
    print("\n2D Convolution with Even Kernel:\n", output_2d_even)
    """
    # Ensure b is a 1D tensor
    if b.ndim != 1:
        raise ValueError("b needs to be a 1-dimensional tensor")

    # Check if `a` is 1D and remember this for later
    is_1d = a.ndim == 1

    # Add a channel dimension to `a` if it's 1D
    if is_1d:
        a = a.unsqueeze(0).unsqueeze(0)
    else:
        a = a.unsqueeze(1)  # Add channel dimension for 2D input

    # Adjust padding based on the length of b
    if len(b) % 2 == 0:  # Even length kernel
        pad_left = len(b) // 2 - 1
        pad_right = len(b) // 2
    else:  # Odd length kernel
        pad_left = pad_right = len(b) // 2

    # Circular padding along the last dimension
    padded_a = F.pad(a, (pad_left, pad_right), mode='circular')

    # Reshape b for convolution
    b = b.view(1, 1, -1)  # (out_channels, in_channels, length)

    # Perform convolution with stride 1 and no additional padding (since we already padded)
    result = F.conv1d(padded_a, b, stride=1)

    # Remove the added channel dimension
    result = result.squeeze(1)

    # Remove the batch dimension if `a` was originally 1D
    return result.squeeze(0) if is_1d else result

def make_blur_conv_kernel_1d(
        d, scale=0.5,
        trunc_end=None,
        dtype=None):
    """
    constructs a blurring 1d convolution kernel of size d
    if trunc_end is not given, center it on the interval.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = torch.linspace(
        -1/scale, 1/scale, d,
        dtype=dtype)
    kernel = torch.exp(-(x**2))
    kernel /= kernel.sum()
    if trunc_end is None:
        return kernel
    return torch.roll(kernel, trunc_end - d // 2, dims=(0,))


def random_top_hats_basis_1d(n, d, dtype=None):
    """
    constructs n random bases in a length-d array.
    hats wrap around the end.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    basis = []
    side_lengths = (torch.rand(n)**2 * (d/2-2) +1).int()
    offsets_x = (torch.rand(n) * d).int().numpy()
    print(side_lengths, offsets_x)
    # I think we could actually vectorize this?
    for i in range(n):
        elem = torch.zeros(d, dtype=dtype)
        elem[
            0:side_lengths[i]
        ] = 1
        elem = elem.roll(offsets_x[i], dims=(0,))
        basis.append(elem)
    return torch.stack(basis)


def random_gaussians_basis_1d(n, d, dtype=None, peak_height=1.0):
    """
    Constructs n random RBF functions in a length-d array.
    All functions will have the same peak height.
    The width of the functions is inversely proportional to n**0.5.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    basis = []
    # Randomize the means and adjust standard deviations
    means = (torch.rand(n) * d)
    # ensure the end has coverage
    means[0] = 0
    means[-1] = d - 1
    std_dev_factor = (d / 2) / (n**0.75)
    std_devs = (torch.rand(n) * std_dev_factor + 1)

    x = torch.arange(d, dtype=dtype)
    for i in range(n):
        # Create a Gaussian function
        gaussian = torch.exp(-0.5 * ((x - means[i]) / std_devs[i])**2)
        gaussian = gaussian * peak_height / gaussian.max()  # Scale to fixed peak height

        basis.append(gaussian)

    return torch.stack(basis)


def random_cyclic_fns_1d(n, d, dtype=None, peak_height=1.0, smoothness=10.0):
    """
    Constructs n random cyclic nearly-RBF functions in a length-d array.
    The width of the peaks is inversely proportional to n**0.5.
    """
    # Scale factor to map indices to phase values
    scale = 2 * torch.pi / d

    # Random central phases phi0 for each vector
    phi0 = torch.rand(n) * 2 * torch.pi
    # Corresponding points x0 on unit circle
    x0 = torch.stack((torch.cos(phi0), torch.sin(phi0)), dim=1)

    # Random widths w for each vector, using a chi-squared distribution with mean mean_w
    mean_w = smoothness * d**-0.5
    w = torch.distributions.Chi2(mean_w).sample((n, 1))

    # Creating a grid of phases for all vectors
    phases = torch.arange(d) * scale
    x = torch.stack((torch.cos(phases), torch.sin(phases)), dim=1)

    # Reshape x to (1, d, 2) and x0 to (n, 1, 2) for correct broadcasting
    x = x[None, :, :]
    x0 = x0[:, None, :]

    # Compute the function values in a vectorized manner
    vectors = torch.exp(-torch.sum((x - x0) ** 2, dim=2) / w)

    return vectors


def gaussian_1d(n):
    # Generate an array of indices centered around 0
    x = torch.linspace(-n / 2.0, n / 2.0, n)

    # Standard deviation is 1/4 of the array length
    std_dev = n / 4.0

    # Compute the PDF of the Gaussian distribution at each index
    gaussian_pdf = torch.exp(-0.5 * (x / std_dev)**2) / (std_dev * torch.sqrt(2 * torch.pi))

    return gaussian_pdf

##
## 2d testers
##

def make_ball_2d(d, radius, trunc_end=(0,0), dtype=None):
    """returns a 2d torch array representing the discretization of a disc of the given radius.
    Values are 1 inside the circle and 0 outside
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = torch.linspace(-d, d, d, dtype=dtype)
    xx, yy = torch.meshgrid(x, x)
    disc = ((xx**2 + yy**2) < radius**2).type(dtype)
    return torch.roll(disc, trunc_end, dims=(0,1))


def channel_pad_2d(a):
    """
    makes an array match the expected format for a 2d conv with 1 channel, which is 4 channels if we don't wish to be ambiguous.
    """
    if a.ndim == 2:
        return a.unsqueeze(0).unsqueeze(0)
    elif a.ndim == 3:
        # there is already a batch dim, need a channel
        return a.unsqueeze(1)


def convolve_array_2d(a, b):
    """convolves two 2d arrays"""
    if dtype is None:
        dtype = torch.get_default_dtype()
    output_ndim = max(a.ndim, b.ndim)
    conved = F.conv2d(
        channel_pad_2d(a),
        channel_pad_2d(b),
        padding='same'
    )
    # purge channels
    conved.squeeze_(1)
    if conved.ndim > output_ndim:
        conved.squeeze_(0)
    return conved


def make_blur_conv_kernel_2d(d, scale=0.5,
        dtype=None):
    """
    constructs a blurring 2d convolution kernel of size d
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    x = torch.linspace(
        -1/scale, 1/scale, d,
        dtype=dtype)
    xx, yy = torch.meshgrid(x, x)
    kernel = torch.exp(-(xx**2 + yy**2))
    return kernel / kernel.sum()


def random_top_hats_basis_2d(n, d, dtype=None):
    """
    constructs n random square bases in a d x d array
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    basis = torch.zeros((n, d, d), dtype=dtype)
    side_lengths = (torch.rand(n)**2 * (d/2-2) +1).int()
    offsets_x = (torch.rand(n) * (d - side_lengths)).int()
    offsets_y = (torch.rand(n) * (d - side_lengths)).int()
    for i in range(n):
        basis[i,
            offsets_x[i]:(offsets_x[i]+side_lengths[i]),
            offsets_y[i]:(offsets_y[i]+side_lengths[i])] = 1
    return basis


def random_gaussians_basis_2d(n, d, dtype=None, peak_height=1.0):
    """
    Constructs n random 2D Gaussian functions in a d x d array.
    All Gaussian functions will have the same peak height.
    The standard deviation of the Gaussians is inversely proportional to n**0.25.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    basis = torch.zeros((n, d, d), dtype=dtype)
    means_x = (torch.rand(n) * d)
    means_y = (torch.rand(n) * d)
    std_dev_factor = (d / 4) / (n**0.25)
    std_devs_x = (torch.rand(n) * std_dev_factor + 1)
    std_devs_y = (torch.rand(n) * std_dev_factor + 1)

    x = torch.arange(d, dtype=dtype).view(-1, 1).repeat(1, d)
    y = torch.arange(d, dtype=dtype).view(1, -1).repeat(d, 1)

    for i in range(n):
        # Create a 2D Gaussian function
        gaussian = torch.exp(-0.5 * (((x - means_x[i]) / std_devs_x[i])**2 + ((y - means_y[i]) / std_devs_y[i])**2))
        gaussian = gaussian * peak_height / gaussian.max()  # Scale to fixed peak height

        basis[i] = gaussian

    return basis


def gaussian_2d(array_size, cov_matrix):
    assert cov_matrix.shape == (2, 2), "Covariance matrix must be 2x2"

    # Generate coordinates
    x = torch.linspace(-array_size / 2.0, array_size / 2.0, array_size)
    y = torch.linspace(-array_size / 2.0, array_size / 2.0, array_size)

    x, y = torch.meshgrid(x, y)
    xy_grid = torch.stack((x, y), dim=-1)  # Shape: (array_size, array_size, 2)

    # Scale the covariance matrix
    scale_factor = (array_size / 4.0)**2
    scaled_cov_matrix = cov_matrix * scale_factor

    # Invert the scaled covariance matrix for PDF computation
    inv_cov_matrix = torch.inverse(torch.tensor(scaled_cov_matrix, dtype=torch.float32))

    # Compute the 2D Gaussian PDF
    diff = xy_grid - torch.tensor([0.0, 0.0])
    exponent = torch.einsum("...i,ij,...j->...", diff, inv_cov_matrix, diff)
    gaussian_pdf = torch.exp(-0.5 * exponent) / (2 * torch.pi * torch.sqrt(torch.det(scaled_cov_matrix)))

    return gaussian_pdf
