import torch.nn as nn
import torch


class OutputClassifier(nn.Module):
    """
    MLP with 3 hidden layers and a dropout layer.
    The activation function used is a ReLU.
    This MLP can classify if a seed is true or false (one seed at a time).

    Args:
        input_shape: number of features for 3 hits (one seed)
        hidden_layers: number of neurons for each hidden layer, in order
        p: dropout coefficient
        activation: activation function (ReLU)

    Returns:
        Wether the seed is true (1) or false (0).
    """

    def __init__(
        self,
        input_shape: int,
        hidden_layers: list[int],
        p: float,
        activation: str,
        threshold: float = 0.5,
        output_shape: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super(OutputClassifier, self).__init__()
        # Keep the config used to build this architecture, needed to rebuild it when loading a checkpoint

        self.activation: torch.nn.Module
        if activation == "ReLU":
            self.activation = torch.nn.ReLU()
        elif activation == "Sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "Tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}, supported options are: ReLU, Sigmoid, Tanh")
        # One Linear layer per entry in hidden_layers, chained input_shape -> hidden_layers[0] -> ... -> hidden_layers[-1]
        layer_dims = [input_shape] + list(hidden_layers)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dims[i], layer_dims[i + 1], device=device) for i in range(len(layer_dims) - 1)]
        )
        self.dropout = torch.nn.Dropout(p)
        self.output_layer = torch.nn.Linear(hidden_layers[-1], output_shape, device=device)
        self.threshold = threshold
        self.device_acc = device

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
            out = self.dropout(out)
        out = self.output_layer(out)
        return out
