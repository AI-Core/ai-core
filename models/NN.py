import torch
class NN(torch.nn.Module):
    def __init__(self, layer_widths):
        super().__init__()
        l = []

        n_layers = len(layer_widths)

        for idx in range(n_layers - 1):
            l.append(torch.nn.Linear(layer_widths[idx], layer_widths[idx+1]))   # add a linear layer
            if idx + 1 != n_layers: # if this is not the last layer ( +1 = zero indexed) (-1 = layer b4 last)
                l.append(torch.nn.ReLU())   # activate
        self.layers = torch.nn.Sequential(
            *l
        )

    def forward(self, x):
        x = self.layers(x)
        return x