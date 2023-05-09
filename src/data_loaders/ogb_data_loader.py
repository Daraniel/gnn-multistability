from pathlib import Path
from typing import Union, Dict

import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from torch import Tensor

from common.exceptions import DataWorkflowException


def get_node_dataset(dataset_name: str, dataset_root: Union[str, Path]) -> NodePropPredDataset:
    """
    get a OGB node dataset from its name
    :param dataset_name: name of the dataset
    :param dataset_root: root folder of the datasets
    :return: dataset object
    """
    dataset = NodePropPredDataset(name=dataset_name, root=dataset_root)
    return dataset


def get_link_dataset(dataset_name: str, dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get a OGB link dataset from its name
    :param dataset_name: name of the dataset
    :param dataset_root: root folder of the datasets
    :return: dataset object
    """
    dataset = PygLinkPropPredDataset(name=dataset_name, root=dataset_root)
    return dataset


def get_ogbn_arxiv(dataset_root: Union[str, Path]) -> NodePropPredDataset:
    """
    get ogbn-arxiv dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_node_dataset('ogbn-arxiv', dataset_root)


def get_ogbn_products(dataset_root: Union[str, Path]) -> NodePropPredDataset:
    """
    get ogbn-products dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_node_dataset('ogbn-products', dataset_root)


def get_ogbl_ddi(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-ddi dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-ddi', dataset_root)


def get_ogbl_citation2(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-citation2 dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-citation2', dataset_root)


def get_ogbl_collab(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-collab dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-collab', dataset_root)


def get_ogbl_ppa(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-ppa dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-ppa', dataset_root)


def split_dataset(dataset: Union[NodePropPredDataset, PygLinkPropPredDataset]) \
        -> Dict[str, Union[np.ndarray, Dict[str, Tensor]]]:
    """
    splits the given dataset to train, valid and test splits
    :param dataset: dictionary of the splits
    """
    if isinstance(dataset, NodePropPredDataset):
        return dataset.get_idx_split()
    elif isinstance(dataset, PygLinkPropPredDataset):
        return dataset.get_edge_split()
    else:
        raise DataWorkflowException(f"Dataset type {type(dataset)} is not supported, only "
                                    f"NodePropPredDataset, PygLinkPropPredDataset datasets are supported")
