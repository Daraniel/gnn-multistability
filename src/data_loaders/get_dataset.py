from pathlib import Path
from typing import Union, Dict, Tuple

import torch_geometric
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

import data_loaders.ogb_data_loader as ogb
import data_loaders.tudataset_data_loader as tu
from common.exceptions import DataWorkflowException

# HINT: when adding a regression dataset, also add the dataset to REGRESSION_DATASETS and if it is a single value
# regression dataset, also add it to SINGLE_VALUE_REGRESSION_DATASETS sets defined in tudataset_data_loader.py file
DATASETS = {
    # graph classification
    # 'imdb_binary': tu.get_imdb_binary, # dataset is disabled because it is missing X values
    'aids': tu.get_aids,
    'enzymes': tu.get_enzymes,
    'ptc_fm': tu.get_ptc_fm,
    'proteins': tu.get_proteins,

    'yeast': tu.get_yeast,  # this dataset is too big and training on it is too slow

    # node classification (may or may not work)
    'arxiv': ogb.get_ogbn_arxiv,
    'products': ogb.get_ogbn_products,

    # link prediction
    'ddi': ogb.get_ogbl_ddi,
    'collab': ogb.get_ogbl_collab,
    'ppa': ogb.get_ogbl_ppa,
    'biokg': ogb.get_ogbl_biokg,
    'citation2': ogb.get_ogbl_citation2,
    'wikikg2': ogb.get_ogbl_wikikg2,

    # graph regression
    'alchemy': tu.get_alchemy,
    'toluene': tu.get_toluene,
    'salicylic_acid': tu.get_salicylic_acid,
    'uracil': tu.get_uracil,

    'aspirin': tu.get_aspirin,
    'qm9': tu.get_qm9,
    'naphthalene': tu.get_naphthalene,
    'zinc': tu.get_zinc,  # this dataset is huge and training on it needs a ton of VRAM or batching

}


def get_dataset(dataset_name: str, dataset_root: Union[str, Path]) \
        -> Union[Dict[str, Dataset], Tuple[Dataset, Dict[str, Tensor]]]:
    """
    get a dataset from its name
    :param dataset_name: name of the dataset
    :param dataset_root: root folder of the datasets
    :return dictionary of the split datasets and a None split (tu dataset)
     or the full dataset and indices of split_edge (ogb dataset)
    """
    dataset_name = dataset_name.lower().replace('-', '_')
    if dataset_name not in DATASETS.keys():
        raise DataWorkflowException(f"Dataset {dataset_name} is not supported")
    dataset = DATASETS[dataset_name](dataset_root)
    return split_dataset(dataset)


def split_dataset(dataset) -> Union[Tuple[Dict[str, Dataset], None], Tuple[Dataset, Dict[str, Tensor]]]:
    """
    split_edge the given dataset to train, valid and test split_edge
    :param dataset: dataset to split
    :return dictionary of the split datasets and a None split (tu dataset)
     or the full dataset and indices of split_edge (ogb dataset)
    """
    if isinstance(dataset, TUDataset) or isinstance(dataset, torch_geometric.datasets.qm9.QM9):
        return tu.split_dataset(dataset), None
    elif isinstance(dataset, NodePropPredDataset) or isinstance(dataset, PygLinkPropPredDataset):
        # raise NotImplementedError("ogb is not implemented yet")  # TODO: implement
        return dataset, ogb.split_dataset(dataset)
        # return ogb.split_dataset(dataset), None
    else:
        raise DataWorkflowException(f"Dataset type {type(dataset)} is not supported")
