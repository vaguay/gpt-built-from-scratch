import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        #embed input batch of sentences padded with tokens of vocab size which get each embedded into 16 dim tokens which we average across the sequence dimension to collapse into input one fixed size vetcor
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor (0, 1, 2)
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        x = self.embedding(x)
        #eplace every token ID with its 16-dimensional embedding.
        #average token vectors so each batch sample gets one vector of 16
        #we want to average over rows
        
        x = x.mean(dim=1)
        #collapse the token dimension, giving one representation per sentence.
        x = self.linear(x)
        #turn each 16-feature sentence representation into one score.
        x = self.sigmoid(x)
        #turn that score into a value between 0 and 1.
        return torch.round(x, decimals=4)
