import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # type: ignore


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

    def forward(self, x):
        # input layer
        x = F.relu(self.layers["input"](x))

        # hidden layers
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[f"hidden_{i}"](x))

        # output layer
        x = self.layers["output"](x)
        return x

    def summary(self, input_size):
        summary(self, input_size=(1, input_size))


# class AnnDiscriminator(nn.Module):
#     def __init__(self, in_features=4, hidden_layers: list[int]= [2,2,], out_features=2):

#         super(AnnDiscriminator, self).__init__()

#         self.layers = nn.ModuleDict()

#         # # input layer
#         # self.layers["input"] = nn.Linear(in_features, hidden_layers[0])
#         # # hidden layers

#         # for i in range(len(hidden_layers) - 1):
#         #     self.layers[f"hidden_{i}"] = nn.Linear(hidden_layers[i], hidden_layers[i + 1])

#         # # output layer
#         # self.layers["output"] = nn.Linear(hidden_layers[-1], out_features)

#         nUnits =4
#         ### input layer
#         self.layers['input'] = nn.Linear(in_features,nUnits)

#         ### hidden layers
#         for i in range(2):
#             self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

#         ### output layer
#         self.layers['output'] = nn.Linear(nUnits,out_features)

#     def forward(self, x):
#         print("x.shape", x.shape)
#         # input layer (note: the code in the video omits the relu after this layer)
#         x = F.relu( self.layers['input'](x) )

#         # hidden layers
#         for i in range(len(self.layers) - 2):
#             x = F.relu( self.layers[f'hidden{i}'](x) )

#         # return output layer
#         x = self.layers['output'](x)

#         return x
