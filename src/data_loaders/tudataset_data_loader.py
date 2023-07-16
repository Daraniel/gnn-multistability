# based on the sample TUDataset code: https://github.com/chrsmrrs/tudataset
from pathlib import Path
from typing import Dict, Union

import torch
import torch_geometric.transforms as transforms
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

from common.exceptions import DataWorkflowException


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


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


def get_qm9(dataset_root: Union[str, Path]) -> Dataset:
    """
    get QM9 dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    dataset_name = 'QM9'
    return get_dataset(dataset_name, dataset_root)


def split_dataset(dataset: TUDataset) -> Dict[str, Dataset]:
    """
    splits the given dataset to train, valid and test splits
    :param dataset: dictionary of the splits
    """
    if not isinstance(dataset, TUDataset):
        raise DataWorkflowException(
            f"Dataset type {type(dataset)} is not supported, only TUDatasett datasets are supported")

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
    # 70 train, 20 val, 10 test
    train_index, test_index = train_test_split(range(len(dataset)), test_size=0.1)
    train_index, val_index = train_test_split(train_index, test_size=0.22)
    train = dataset[train_index]
    valid = dataset[val_index]
    test = dataset[test_index]
    return {'train': train, 'valid': valid, 'test': test}
