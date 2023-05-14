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
        self.conv1 = GINConv(Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            BN(hidden_dim),
        ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    BN(hidden_dim),
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
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


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
