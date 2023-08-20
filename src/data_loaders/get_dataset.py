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
    # graph classification
    # 'imdb_binary': tu.get_imdb_binary,  # dataset is disabled because it is missing X values
    'aids': tu.get_aids,
    'enzymes': tu.get_enzymes,
    'ptc_fm': tu.get_ptc_fm,
    'proteins': tu.get_proteins,

    'yeast': tu.get_yeast,  # this dataset is too big and training on it is too slow

    'arxiv': ogb.get_ogbn_arxiv,
    'products': ogb.get_ogbn_products,
    'ddi': ogb.get_ogbl_ddi,
    'citation2': ogb.get_ogbl_citation2,
    'collab': ogb.get_ogbl_collab,
    'ppa': ogb.get_ogbl_ppa,

    # graph regression
    'alchemy': tu.get_alchemy,
    'aspirin': tu.get_aspirin,
    'qm9': tu.get_qm9,
    'naphthalene': tu.get_naphthalene,

    'salicylic_acid': tu.get_salicylic_acid,
    'zinc': tu.get_zinc,  # this dataset is huge and training on it needs a ton of VRAM or batching

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
