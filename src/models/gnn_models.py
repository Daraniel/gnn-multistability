from collections import OrderedDict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
# noinspection PyProtectedMember
from torch_geometric.nn import GINConv, global_mean_pool


# based on https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
class GIN(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float, **kwargs):
        super(GIN, self).__init__()
        # self.conv1 = GINConv(ModuleDict(
        #     {
        #         "lin1": Linear(in_dim, hidden_dim),
        #         "act1": ReLU(),
        #         "lin2": Linear(hidden_dim, hidden_dim),
        #         "act2": ReLU(),
        #         "bn": BN(hidden_dim),
        #     }
        # ),
        #     train_eps=True)

        self.convs = torch.nn.ModuleList()

        self.convs.append(GINConv(Sequential(
            OrderedDict([
                ("lin1", Linear(in_dim, hidden_dim)),
                ("act1", ReLU()),
                ("lin2", Linear(hidden_dim, hidden_dim)),
                ("act2", ReLU()),
                ("bn", BN(hidden_dim)),
            ])
        ),
            train_eps=True)
        )

        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    OrderedDict([
                        ("lin1", Linear(hidden_dim, hidden_dim)),
                        ("act1", ReLU()),
                        ("lin2", Linear(hidden_dim, hidden_dim)),
                        ("act2", ReLU()),
                        ("bn", BN(hidden_dim)),
                    ])
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

    def activations(self, data):
        """
        gets the values of the activation functions for the given input
        :param data: the input dateset
        :return: the value of the activation functions
        """
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        # for i, layer in enumerate(self.convs[:-1], start=1):
        for i, layer in enumerate(self.convs):
            # go over the internal layers of GIN to get the value of its activation last function
            temp_x = layer.nn.lin1(x)
            temp_x = layer.nn.act1(temp_x)
            temp_x = layer.nn.lin2(temp_x)
            temp_x = layer.nn.act2(temp_x)
            hs[f"{i}.0"] = temp_x  # HINT: save the value of the second activation function of each GIN layer

            # x = layer["bn"](x)  # HINT: we can ignore the value of the batch normalization since it's not needed
            x = layer(x, edge_index)  # HINT: get the result of the whole GIN layer, so we can feed it to the next layer
        hs[f"{len(self.convs)}.0"] = F.relu(self.lin1(x))

        return hs


MODELS = {
    'gin': GIN,
}


def get_model(cfg: DictConfig, in_dim: int, out_dim: int) -> torch.nn.Module:
    """
    get a model object from the config
    :param cfg: project configuration
    :param in_dim: input dimensions
    :param out_dim: output dimensions
    :return: specified model object
    """
    return MODELS[cfg.model.name](in_dim=in_dim, out_dim=out_dim, **cfg.model)
