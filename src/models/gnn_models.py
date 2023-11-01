import abc
from collections import OrderedDict
from typing import Dict, Type

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, GRU
# noinspection PyProtectedMember
from torch_geometric.nn import GINConv, global_mean_pool, GATConv, GCNConv, SAGEConv, GatedGraphConv, ResGatedGraphConv, \
    NNConv, Set2Set

from common.utils import TaskType


# this class is based on https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


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
                 task_type: TaskType, **kwargs):
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
        self.task_type = task_type

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
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
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)

        else:
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
                 n_heads: int, n_output_heads: int, task_type: TaskType, **kwargs) -> None:
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)
        else:
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
                 task_type: TaskType, **kwargs) -> None:
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)

        else:
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
    def __init__(self, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float, task_type: TaskType,
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)
        else:
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
                 task_type: TaskType, **kwargs) -> None:
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)

        else:
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
                 task_type: TaskType, **kwargs) -> None:
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)

        else:
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
                 task_type: TaskType, **kwargs) -> None:
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
        self.task_type = task_type

    def forward(self, data):
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index, batch = data.x, data.adj_t, data.batch
            else:
                x, edge_index, batch = data.emb.weight, data.adj_t, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore

        if self.task_type == TaskType.LINK_PREDICTION:
            return x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        if self.task_type == TaskType.REGRESSION:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def activations(self, data):
        hs = {}
        device = next(self.parameters()).device
        if self.task_type == TaskType.LINK_PREDICTION:
            if hasattr(data, 'x') and data.x is not None:
                x, edge_index = data.x.to(device), data.adj_t.to(device)
            else:
                x, edge_index = data.emb.weight.to(device), data.adj_t.to(device)

        else:
            x, edge_index = data.x.to(device), data.edge_index.to(device)
        for i, layer in enumerate(self.layers[:-1], start=1):  # type:ignore
            x = layer["dropout"](x)  # type:ignore
            x = layer["conv"](x, edge_index)  # type:ignore
            if "act" in layer:  # type:ignore
                x = layer["act"](x)  # type:ignore
            hs[f"{i}.0"] = x

        hs[f"{len(self.layers)}.0"] = F.relu(self.lin1(x))
        return hs


# based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
class MPNN(GNNBaseModel):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
                 task_type: TaskType, **kwargs) -> None:
        super(MPNN, self).__init__()
        self.lin0 = torch.nn.Linear(in_dim, hidden_dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)

        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_dim)
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(self.num_layers):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

    def activations(self, data) -> Dict[str, torch.Tensor]:
        hs = {}
        device = next(self.parameters()).device
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)

        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(self.num_layers):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            hs[f"{i}.0"] = out

        # out = self.set2set(out, data.batch)
        # hs[f"{self.num_layers}.0"] = out
        # out = F.relu(self.lin1(out))
        # out = self.lin2(out)
        return hs


# # based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
# class DropMPNN(torch.nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, dropout_p: float,
#                  task_type: TaskType, **kwargs) -> None:
#         super(DropMPNN, self).__init__()
#         self.lin0 = torch.nn.Linear(in_dim, hidden_dim)
#
#         nn = Sequential(Linear(5, 128), ReLU(), Linear(128, hidden_dim * hidden_dim))
#         self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
#         self.gru = GRU(hidden_dim, hidden_dim)
#
#         self.set2set = Set2Set(hidden_dim, processing_steps=3)
#         self.lin1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, 1)
#
#         self.num_layers = num_layers
#         self.num_runs = 1
#         self.dropout_p = dropout_p
#         # if args.aux_loss:
#         #     self.set2set_aux = Set2Set(hidden_dim, processing_steps=3)
#         #     self.lin1_aux = torch.nn.Linear(2 * hidden_dim, hidden_dim)
#         #     self.lin2_aux = torch.nn.Linear(hidden_dim, 1)
#
#     def forward(self, data):
#         device = next(self.parameters()).device
#         aux_x = None
#         # Do runs in paralel, by repeating the graphs in the batch
#         x = data.x
#         x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
#         drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=device) * self.dropout_p).bool()
#         x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=device)
#         del drop
#         x = x.view(-1, x.size(-1))
#         run_edge_index = data.edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs,
#                                                                                  device=device).repeat_interleave(
#             data.edge_index.size(1)) * (data.edge_index.max() + 1)
#         run_edge_attr = data.edge_attr.repeat(self.num_runs, 1)
#         x = F.relu(self.lin0(x))
#         h = x.unsqueeze(0)
#         for i in range(self.num_layers):
#             m = F.relu(self.conv(x, run_edge_index, run_edge_attr))
#             x, h = self.gru(m.unsqueeze(0), h)
#             x = x.squeeze(0)
#         del run_edge_index, run_edge_attr
#         # if args.aux_loss:
#         #     run_batch = data.batch.repeat(self.num_runs) + torch.arange(self.num_runs, device=device).repeat_interleave(data.batch.size(0)) * (data.batch.max() + 1)
#         #     aux_x = self.set2set_aux(x, run_batch)
#         #     del run_batch
#         #     aux_x = F.relu(self.lin1_aux(aux_x))
#         #     aux_x = self.lin2_aux(aux_x)
#         #     aux_x = aux_x.view(self.num_runs, -1, aux_x.size(-1))
#         x = x.view(self.num_runs, -1, x.size(-1))
#         x = x.mean(dim=0)
#
#         x = self.set2set(x, data.batch)
#         x = F.relu(self.lin1(x))
#         x = self.lin2(x)
#         return x.view(-1)
#
#     def activations(self, data) -> Dict[str, torch.Tensor]:
#         hs = {}
#         device = next(self.parameters()).device
#         aux_x = None
#         # Do runs in paralel, by repeating the graphs in the batch
#         x = data.x.to(device)
#         x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
#         edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
#         drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=device) * self.dropout_p).bool()
#         x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=device)
#         del drop
#         x = x.view(-1, x.size(-1))
#         run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs,
#                                                                             device=device).repeat_interleave(
#             edge_index.size(1)) * (edge_index.max() + 1)
#         run_edge_attr = edge_attr.repeat(self.num_runs, 1)
#         x = F.relu(self.lin0(x))
#         h = x.unsqueeze(0)
#         for i in range(self.num_layers):
#             m = F.relu(self.conv(x, run_edge_index, run_edge_attr))
#             x, h = self.gru(m.unsqueeze(0), h)
#             x = x.squeeze(0)
#             hs[f"{i}.0"] = x
#
#         del run_edge_index, run_edge_attr
#         # if args.aux_loss:
#         #     run_batch = data.batch.repeat(self.num_runs) + torch.arange(self.num_runs, device=device).repeat_interleave(data.batch.size(0)) * (data.batch.max() + 1)
#         #     aux_x = self.set2set_aux(x, run_batch)
#         #     del run_batch
#         #     aux_x = F.relu(self.lin1_aux(aux_x))
#         #     aux_x = self.lin2_aux(aux_x)
#         #     aux_x = aux_x.view(self.num_runs, -1, aux_x.size(-1))
#         x = x.view(self.num_runs, -1, x.size(-1))
#         x = x.mean(dim=0)
#
#         x = self.set2set(x, data.batch)
#         hs[f"{self.num_layers}.0"] = x
#         return hs


MODELS: Dict[str, Type[GNNBaseModel]] = {
    'gin': GIN,
    'gat': GAT2017,
    'gcn': GCN2017,
    'gatedgcn': GatedGCN,
    'graphsage': GraphSAGE,
    'resgatedgcn': ResGatedGCN,
    'mpnn': MPNN,
    # 'drop_mpnn': DropMPNN,
}


def get_model(cfg: DictConfig, in_dim: int, out_dim: int, task_type: TaskType) -> GNNBaseModel:
    """
    get a model object from the config
    :param cfg: project configuration
    :param in_dim: input dimensions
    :param out_dim: output dimensions
    :param task_type: whether task is regression or classification (if false, results will pass through softmax)
    :return: specified model object
    """
    return MODELS[cfg.model.name](in_dim=in_dim, out_dim=out_dim, task_type=task_type, **cfg.model)
