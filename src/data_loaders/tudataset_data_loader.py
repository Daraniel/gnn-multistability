# based on the sample TUDataset code: https://github.com/chrsmrrs/tudataset
from pathlib import Path
from typing import Dict, Union

import torch
import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.transforms as transforms
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.utils import degree, remove_self_loops

from common.exceptions import DataWorkflowException

CLASSIFICATION_DATASETS = {'aids', 'enzymes', 'ptc_fm', 'proteins', 'yeast'}
REGRESSION_DATASETS = {'alchemy', 'aspirin', 'qm9', 'toluene', 'naphthalene', 'salicylic_acid', 'zinc', 'uracil'}
SINGLE_VALUE_REGRESSION_DATASETS = {'aspirin', 'qm9', 'toluene', 'naphthalene', 'salicylic_acid', 'zinc_full', 'uracil'}


def get_dataset(dataset_name, dataset_root: Union[str, Path]) -> Dataset:
    """
    get a TUDataset from its name
    :param dataset_name: name of the dataset
    :param dataset_root: root folder of the datasets
    :return: dataset object
    """
    try:
        dataset = TUDataset(dataset_root, name=dataset_name, use_node_attr=False, use_edge_attr=False, cleaned=False)
        return dataset
    except Exception as e:
        raise DataWorkflowException(f'Failed to get dataset {dataset_name}') from e


def get_imdb_binary(dataset_root: Union[str, Path]) -> Dataset:
    """
    get IMDB-BINARY dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'IMDB-BINARY'
    return get_dataset(dataset_name, dataset_root)


def get_aids(dataset_root: Union[str, Path]) -> Dataset:
    """
    get AIDS dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'AIDS'
    return get_dataset(dataset_name, dataset_root)


def get_proteins(dataset_root: Union[str, Path]) -> Dataset:
    """
    get PROTEINS dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'PROTEINS'
    return get_dataset(dataset_name, dataset_root)


def get_enzymes(dataset_root: Union[str, Path]) -> Dataset:
    """
    get ENZYMES dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'ENZYMES'
    return get_dataset(dataset_name, dataset_root)


def get_yeast(dataset_root: Union[str, Path]) -> Dataset:
    """
    get YEAST dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'Yeast'
    return get_dataset(dataset_name, dataset_root)


def get_ptc_fm(dataset_root: Union[str, Path]) -> Dataset:
    """
    get PTC_FM dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'PTC_FM'
    return get_dataset(dataset_name, dataset_root)


def get_zinc(dataset_root: Union[str, Path]) -> Dataset:
    """
    get ZINC dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'ZINC_full'
    return get_dataset(dataset_name, dataset_root)


def get_alchemy(dataset_root: Union[str, Path]) -> Dataset:
    """
    get alchemy dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'alchemy_full'
    return get_dataset(dataset_name, dataset_root)


def get_aspirin(dataset_root: Union[str, Path]) -> Dataset:
    """
    get aspirin dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'aspirin'
    return get_dataset(dataset_name, dataset_root)


# this function is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# this function is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, 1]
        return data


# this function is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


# this function is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
def get_qm9(dataset_root: Union[str, Path]) -> Dataset:
    """
    get QM9 dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'QM9'
    # transform = T.Compose([MyTransform(), Complete()])
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    # transform = T.Compose([MyTransform(), T.Distance(norm=False)])
    # transform = T.Compose([MyTransform()])
    dataset = QM9(dataset_root, transform=transform).shuffle()
    dataset.name = dataset_name
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    return dataset
    # return get_dataset(dataset_name, dataset_root)


def get_toluene(dataset_root: Union[str, Path]) -> Dataset:
    """
    get toluene dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'toluene'
    return get_dataset(dataset_name, dataset_root)


def get_naphthalene(dataset_root: Union[str, Path]) -> Dataset:
    """
    get naphthalene dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'naphthalene'
    return get_dataset(dataset_name, dataset_root)


def get_salicylic_acid(dataset_root: Union[str, Path]) -> Dataset:
    """
    get salicylic_acid dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'salicylic_acid'
    return get_dataset(dataset_name, dataset_root)


def get_uracil(dataset_root: Union[str, Path]) -> Dataset:
    """
    get uracil dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'uracil'
    return get_dataset(dataset_name, dataset_root)


def split_dataset(dataset: TUDataset) -> Dict[str, Dataset]:
    """
    split_edge the given dataset to train, valid and test split_edge
    :param dataset: dictionary of the split_edge
    """
    if not(isinstance(dataset, TUDataset) or isinstance(dataset, torch_geometric.datasets.qm9.QM9)):
        raise DataWorkflowException(
            f"Dataset type {type(dataset)} is not supported, only TUDataset datasets and qm9 are supported")

    # One-hot degree if node labels are not available.
    # Following is taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = transforms.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    dataset_length = len(dataset)
    if dataset.name in REGRESSION_DATASETS:
        # 70% train, 5000 instances test, rest for val, this is done since regression datasets are too big and we
        # want fast evaluation
        test_size = 5000
        train_size = dataset_length * 70 // 100
        dataset_length_after_test = dataset_length - test_size
        valid_size = (dataset_length_after_test - train_size) / dataset_length_after_test
        train_index, test_index = train_test_split(range(dataset_length), test_size=test_size / dataset_length)
        train_index, val_index = train_test_split(train_index, test_size=valid_size)
    else:
        # 70% train, 20% val, 10% test
        train_index, test_index = train_test_split(range(dataset_length), test_size=0.1)
        train_index, val_index = train_test_split(train_index, test_size=0.22)
    train = dataset[train_index]
    valid = dataset[val_index]
    test = dataset[test_index]
    return {'train': train, 'valid': valid, 'test': test}
