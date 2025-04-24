import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary  # type: ignore
import torch.autograd.profiler as profiler

class AnnDiscriminator(nn.Module):
    def __init__(self, model_paramaters):
        super(AnnDiscriminator, self).__init__()

        self.layers = nn.ModuleDict()
        # input layer
        self.layers["input"] = nn.Linear(
            model_paramaters["in_features"], model_paramaters["hidden_layers"][0]
        )

        # hidden layers
        for i in range(len(model_paramaters["hidden_layers"]) - 1):
            self.layers[f"hidden_{i}"] = nn.Linear(
                model_paramaters["hidden_layers"][i],
                model_paramaters["hidden_layers"][i + 1],
            )

        self.layers["output"] = nn.Linear(
            model_paramaters["hidden_layers"][-1], model_paramaters["out_features"]
        )
        pass
    
    
    def forward_profiler(self, x, mask):
        with profiler.record_function("LINEAR PASS"):
            x = self.forward(x)
    

        with profiler.record_function("MASK INDICES"):
            threshold = x.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return x, hi_idx
        
    def forward(self, x):
        # input layer
        x = F.relu(self.layers["input"](x))

        # hidden layers
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[f"hidden_{i}"](x))

        # output layer
        x = self.layers["output"](x)
        return x

    # def summary(self, input_size):
    #     summary(self, input_size=(1, input_size))

    def reset(self):
        for layer in self.layers:
            if hasattr(self.layers[layer], "reset_parameters"):
                self.layers[layer].reset_parameters()
                
