from pathlib import Path
from typing import Union, Dict

import torch_geometric.transforms as T
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
    :param transform: transformation to apply to dataset
    :return: dataset object
    """
    dataset = PygLinkPropPredDataset(name=dataset_name, root=dataset_root, transform=T.ToSparseTensor())
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


def get_ogbl_wikikg2(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-citation2 dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-wikikg2', dataset_root)


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


def get_ogbl_biokg(dataset_root: Union[str, Path]) -> PygLinkPropPredDataset:
    """
    get ogbl-biokg dataset
    :param dataset_root: root folder of the datasets
    :return: the dataset object
    """
    return get_link_dataset('ogbl-biokg', dataset_root)


def split_dataset(dataset: Union[NodePropPredDataset, PygLinkPropPredDataset]) -> Dict[str, Tensor]:
    """
    split_edge the given dataset to train, valid and test split_edge
    :param dataset: dictionary of the split_edge
    """
    if isinstance(dataset, NodePropPredDataset):
        # idx: Dict[str, torch.Tensor] = dataset.get_edge_split()  # type:ignore
        return dataset.get_idx_split()
    elif isinstance(dataset, PygLinkPropPredDataset):
        # idx: Dict[str, torch.Tensor] = dataset.get_edge_split()  # type:ignore
        return dataset.get_edge_split()
    else:
        raise DataWorkflowException(f"Dataset type {type(dataset)} is not supported, only "
                                    f"NodePropPredDataset, PygLinkPropPredDataset datasets are supported")
    # # convert each index of variable length with node ids into a boolean vector with fixed length
    # for key, tensor in idx.items():
    #     new_tensor = torch.zeros((dataset.num_nodes,), dtype=torch.bool)  # type:ignore
    #     new_tensor[tensor] = True
    #     idx[key] = new_tensor
    # return idx
