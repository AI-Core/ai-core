
import torch

class VerboseLayer(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, x):
        print(f'layer: {self.layer_idx}\tshape: {x.shape}')
        return x