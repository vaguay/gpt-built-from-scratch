import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        collect = set()
        key = {}
        final = []
        for sentence in positive:
            for word in sentence.split():
                collect.add(word)
        for sentence in negative:
            for word in sentence.split():
                collect.add(word)

        collect = sorted(collect)

        for i, word in enumerate(collect, start=1):
            key[word] = i

        for sentence in positive:
            encoded = []
            for word in sentence.split():
                encoded.append(key[word])
            final.append(torch.tensor(encoded))

        for sentence in negative:
            encoded = []
            for word in sentence.split():
                encoded.append(key[word])
            final.append(torch.tensor(encoded))
        
        final = nn.utils.rnn.pad_sequence(final, batch_first=True)

        return final



    
    





