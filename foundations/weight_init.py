import torch
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))

        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights * 10000).div(10000).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / fan_in)

        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights * 10000).div(10000).tolist()


    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        weights = []

        # Build all layer weights first
        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))

            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)

            else:  # random N(0, 1)
                std = 1.0

            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)

        # Generate input after weights
        x = torch.randn(1, input_dim)

        activation_stds = []

        for W in weights:
            # Linear layer
            x = x @ W.T

            # ReLU
            x = torch.relu(x)

            activation_stds.append(
                round(x.std().item(), 2)
            )

        return activation_stds