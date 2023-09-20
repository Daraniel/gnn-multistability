import abc
from collections import OrderedDict
from typing import Dict, Type

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
# noinspection PyProtectedMember
from torch_geometric.nn import GINConv, global_mean_pool, GATConv, GCNConv, SAGEConv, GatedGraphConv, ResGatedGraphConv


class GNNBaseModel(torch.nn.Module, abc.ABC):
    # @abc.abstractmethod
    # def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float, *args, **kwargs):
    #     pass

    @abc.abstractmethod
    def forward(self, data):
        pass

    @abc.abstractmethod
    def activations(self, data) -> Dict[str, torch.Tensor]:
        pass


# based on https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
class GIN(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 is_regression: bool, **kwargs):
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

        for i in range(num_layers - 1):  # 1 other layers Conv
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
        self.is_regression = is_regression

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
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

    def activations(self, data) -> Dict[str, torch.Tensor]:
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

            # x = layer["bn"](x) # HINT: we can ignore the value of the batch normalization since it's not needed
            x = layer(x, edge_index)  # HINT: get the result of the whole GIN layer, so we can feed it to the next layer
        hs[f"{len(self.convs)}.0"] = F.relu(self.lin1(x))

        return hs


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/models/gnn.py
class GAT2017(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 n_heads: int, n_output_heads: int, is_regression: bool, **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        in_dim, hidden_dim, heads=n_heads, dropout=dropout_p,
                    ),
                    "act": nn.ELU(),
                }
            )
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GATConv(
                            hidden_dim * n_heads,
                            hidden_dim,
                            heads=n_heads,
                            dropout=dropout_p,
                        ),
                        "act": nn.ELU(),
                    }
                )
            )
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GATConv(
                        hidden_dim * n_heads,
                        hidden_dim,
                        heads=n_output_heads,
                        dropout=dropout_p,
                        concat=False,
                    ),
                }
            )
        )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p
        self.is_regression = is_regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/models/gnn.py
class GCN2017(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 is_regression: bool, **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": GCNConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(num_layers - 1):  # 1 other layers Conv
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": GCNConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


class GatedGCN(GNNBaseModel):
    def __init__(self, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float, is_regression: bool,
                 **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        for _ in range(num_layers):  # 1 other layers Conv
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        # "conv": GatedGraphConv(hidden_dim, hidden_dim),
                        "conv": GatedGraphConv(num_layers=1, out_channels=hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p
        self.is_regression = is_regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


class GraphSAGE(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 is_regression: bool, **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": SAGEConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(num_layers - 1):  # 1 other layers Conv
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": SAGEConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p
        self.is_regression = is_regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


class ResGatedGCN(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 is_regression: bool, **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": ResGatedGraphConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(num_layers - 1):  # 1 other layers Conv
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": ResGatedGraphConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p
        self.is_regression = is_regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


class ResGatedGCNs(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 is_regression: bool, **kwargs) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"n_layers must be larger or equal to 2: {num_layers=}")

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleDict(
                {
                    "dropout": nn.Dropout(dropout_p),
                    "conv": ResGatedGraphConv(in_dim, hidden_dim),
                    "act": nn.ReLU(),
                }
            )
        )
        for _ in range(num_layers - 1):  # 1 other layers Conv
            self.layers.append(
                nn.ModuleDict(
                    {
                        "dropout": nn.Dropout(dropout_p),
                        "conv": ResGatedGraphConv(hidden_dim, hidden_dim),
                        "act": nn.ReLU(),
                    }
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout_p = dropout_p
        self.is_regression = is_regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.is_regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


MODELS: Dict[str, Type[GNNBaseModel]] = {
    'gin': GIN,
    'gat': GAT2017,
    'gcn': GCN2017,
    'gatedgcn': GatedGCN,
    'graphsage': GraphSAGE,
    'resgatedgcn': ResGatedGCN,
}


def get_model(cfg: DictConfig, in_dim: int, out_dim: int, is_regression: bool) -> GNNBaseModel:
    """
    get a model object from the config
    :param cfg: project configuration
    :param in_dim: input dimensions
    :param out_dim: output dimensions
    :param is_regression: whether task is regression or classification (if false, results will pass through softmax)
    :return: specified model object
    """
    return MODELS[cfg.model.name](in_dim=in_dim, out_dim=out_dim, is_regression=is_regression, **cfg.model)
