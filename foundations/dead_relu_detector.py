import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch bc batches are rows of samples and cols are neurones
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead = []
        curr = x

        with torch.no_grad():
            #layer is module and curr is tensor
            for layer in model:
                curr = layer(curr)
                if isinstance(layer, nn.ReLU):
                    dead_neurons = (curr==0).all(dim=0)
                    #torch.tensor([True, True, True]).all()
                    dead_fraction = dead_neurons.float().mean().item()
                    #to python numnber
                    dead.append(round(dead_fraction, 4))
        return dead




    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        # 1. any layer > 0.5
        if any(x > 0.5 for x in dead_fractions):
            return "use_leaky_relu"

        # 2. first layer > 0.3
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. strictly increases with depth AND last > 0.1
        inc = True

        for i in range(len(dead_fractions) - 1):
            if dead_fractions[i + 1] <= dead_fractions[i]:
                inc = False
                break

        if inc and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        # 4. max < 0.1
        if max(dead_fractions) < 0.1:
            return "healthy"

        # 5. otherwise
        return "healthy"
