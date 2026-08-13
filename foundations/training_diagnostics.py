import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        model.train()
        with torch.no_grad():
            current = x
            for layer in model:
                current = layer(current)

                if isinstance(layer, nn.Linear):
                    mean = current.mean().item()
                    std = current.std().item()
                    dead_neurons = (current <= 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean().item()
                    stats.append({
        "mean": round(mean, 4),
        "std": round(std, 4),
        "dead_fraction": round(dead_fraction, 4) })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.train()
        model.zero_grad()

        predictions = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)
        loss.backward() #loss grad with respect to model param
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                mean = grad.mean().item()
                std = grad.std().item()
                norm = grad.norm().item()

                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })

        return stats
        
                    

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for act_stat in activation_stats:
            if act_stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        
        for grad_stat in gradient_stats:
            if grad_stat["norm"] > 1000:
                return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for act_stat in activation_stats:
            if act_stat["std"] < 0.1:
                return "vanishing_gradients"

            if act_stat["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
