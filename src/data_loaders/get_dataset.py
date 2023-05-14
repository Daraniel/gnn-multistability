from pathlib import Path
from typing import Union, Dict

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

import data_loaders.ogb_data_loader as ogb
import data_loaders.tudataset_data_loader as tu
from common.exceptions import DataWorkflowException

DATASETS = {
    'arxiv': ogb.get_ogbn_arxiv,
    'products': ogb.get_ogbn_products,
    'ddi': ogb.get_ogbl_ddi,
    'citation2': ogb.get_ogbl_citation2,
    'collab': ogb.get_ogbl_collab,
    'ppa': ogb.get_ogbl_ppa,
    'imdb_binary': tu.get_imdb_binary,
    'proteins': tu.get_proteins,
    'enzymes': tu.get_enzymes,
    'alchemy': tu.get_alchemy,
    'zinc': tu.get_zinc,
    'qm9': tu.get_qm9,
}


def get_dataset(dataset_name: str, dataset_root: Union[str, Path]) -> Dict[str, Dataset]:
    """
    get a dataset from its name
    :param dataset_name: name of the dataset
    :param dataset_root: root folder of the datasets
    :return: dataset object
    """
    dataset_name = dataset_name.lower().replace('-', '_')
    if dataset_name not in DATASETS.keys():
        raise DataWorkflowException(f"Dataset {dataset_name} is not supported")
    dataset = DATASETS[dataset_name](dataset_root)
    return split_dataset(dataset)


def split_dataset(dataset) -> Dict[str, Dataset]:
    """
    splits the given dataset to train, valid and test splits
    :param dataset: dictionary of the splits
    """
    if isinstance(dataset, TUDataset):
        return tu.split_dataset(dataset)
    elif isinstance(dataset, NodePropPredDataset) or isinstance(dataset, PygLinkPropPredDataset):
        raise NotImplementedError("ogb is not implemented yet")  # TODO: implement
        # return ogb.split_dataset(dataset)
    else:
        raise DataWorkflowException(f"Dataset type {type(dataset)} is not supported")
