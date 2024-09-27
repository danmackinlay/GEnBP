import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict

class EnsembleAdaptor(torch.nn.Module):
    def __init__(self, model):
        super(EnsembleAdaptor, self).__init__()
        self.model = model
        # Extract and store parameter shapes and names
        params = OrderedDict(self.model.named_parameters())
        self.param_shapes = []
        self.param_sizes = []
        self.param_names = []
        for name, param in params.items():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
            self.param_names.append(name)
        self.total_param_size = sum(self.param_sizes)
        # Placeholder for input and output shapes
        self.input_shape = None
        self.output_shape = None

    def forward(self, x_flat, ensemble_weights):
        """
        x_flat: Tensor of shape (batch_size, input_flat_dim)
        ensemble_weights: Tensor of shape (batch_size, total_param_size)
        """
        batch_size = x_flat.shape[0]
        # Unflatten inputs
        x = self.unflatten_input(x_flat)
        outputs = []
        for i in range(batch_size):
            # Extract per-example parameters
            flat_params_i = ensemble_weights[i]
            params_i = self.unflatten_params(flat_params_i)
            # Compute output with functional call
            y_i = self.functional_forward(params_i, x[i].unsqueeze(0))
            outputs.append(y_i)
        # Concatenate outputs and flatten
        outputs = torch.cat(outputs, dim=0)
        y_flat = self.flatten_output(outputs)
        return y_flat

    def unflatten_params(self, flat_params):
        """
        Convert a flat parameter vector into a dictionary of parameter tensors.
        """
        params = {}
        idx = 0
        for name, size, shape in zip(self.param_names, self.param_sizes, self.param_shapes):
            param = flat_params[idx:idx+size].view(shape)
            params[name] = param
            idx += size
        return params

    def functional_forward(self, params, x):
        """
        Perform a forward pass using the given parameters.
        """
        return functional_call(self.model, params, x)

    def flatten_input(self, x):
        """
        Flatten the input tensor.
        """
        self.input_shape = x.shape[1:]  # Save the input shape for unflattening
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

    def unflatten_input(self, x_flat):
        """
        Unflatten the input tensor to its original shape.
        """
        if self.input_shape is None:
            raise ValueError("Input shape not set. Please call flatten_input first.")
        batch_size = x_flat.shape[0]
        return x_flat.view(batch_size, *self.input_shape)

    def flatten_output(self, y):
        """
        Flatten the output tensor.
        """
        self.output_shape = y.shape[1:]  # Save the output shape for unflattening
        batch_size = y.shape[0]
        return y.view(batch_size, -1)

    def unflatten_output(self, y_flat):
        """
        Unflatten the output tensor to its original shape.
        """
        if self.output_shape is None:
            raise ValueError("Output shape not set. Please call flatten_output first.")
        batch_size = y_flat.shape[0]
        return y_flat.view(batch_size, *self.output_shape)

if __name__ == "__main__":
    # Example usage
    import torch
    import torch.nn as nn
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    from torch.nn.utils.stateless import functional_call
    from collections import OrderedDict

    # Define the model
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(6 * 29 * 29, 120)
            self.fc2 = nn.Linear(120, 10)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = x.view(-1, 6 * 29 * 29)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define the EnsembleAdaptor class (same as above)

    # Instantiate the model and adaptor
    model = MyNet()
    ensemble_nn = EnsembleAdaptor(model)

    # Prepare input data
    batch_size = 5
    channels, height, width = 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # Flatten inputs
    x_flat = ensemble_nn.flatten_input(x)
    print(f"Flattened x shape: {x_flat.shape}")

    # Prepare ensemble weights
    ensemble_weights = torch.randn(batch_size, ensemble_nn.total_param_size)

    # Perform forward pass
    ensemble_y_flat = ensemble_nn(x_flat, ensemble_weights)
    print(f"Flattened output shape: {ensemble_y_flat.shape}")

    # Unflatten outputs
    ensemble_y = ensemble_nn.unflatten_output(ensemble_y_flat)
    print(f"Unflattened y shape: {ensemble_y.shape}")
