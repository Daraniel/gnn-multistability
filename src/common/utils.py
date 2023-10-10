import logging
import os
from enum import Enum
from pathlib import Path
from typing import Union, Any, Dict, Tuple, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from torch import Tensor
from torch_geometric.data import Dataset

from data_loaders.get_dataset import get_dataset
from data_loaders.tudataset_data_loader import REGRESSION_DATASETS, CLASSIFICATION_DATASETS


class TaskType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1
    LINK_PREDICTION = 2


def setup_project(cfg: DictConfig, activations_root: Optional[Union[str, Path]], logger: logging.Logger,
                  make_directories: bool = True) \
        -> Tuple[Union[str, Any], Union[Dict[str, Dataset], Dataset], Path, Path, Path, TaskType, Union[
            None, Dict[str, Tensor]]]:
    """
    set up the project and handle the directories
    :param cfg: project configuration
    :param activations_root: path to store the activations
    :param logger: project logger
    :param make_directories: whether to create the directories or not
    :return: tuple of activations, dataset, figures, predictions, cka_dir directories, task type, and dataset split_edge
    """
    logger.info("Configuring project")
    print(OmegaConf.to_yaml(cfg))
    activations_root, dataset_dir, figures_dir, predictions_dir, cka_dir = get_directories(cfg, activations_root,
                                                                                           make_directories)
    fix_seeds(cfg.datasplit_seed)
    # TODO: update

    # if cfg.proportional_split and cfg.degree_split:
    #     raise ValueError("Only one of proportional_split and degree_split can be true.")
    # if cfg.proportional_split:
    #     split_type = "proportional"
    # elif cfg.degree_split:
    #     split_type = "degree"
    # else:
    #     split_type = "num"
    # dataset = get_dataset(
    #     name=cfg.dataset.name,
    #     root=dataset_dir,
    #     transforms=[T.ToSparseTensor(remove_edge_index=False)],
    #     public_split=cfg.public_split,
    #     split_type=split_type,
    #     num_train_per_class=cfg.num_train_per_class,
    #     part_val=cfg.part_val,
    #     part_test=cfg.part_test,
    # )
    dataset_name = get_dataset_name(cfg)
    logger.info(f"Loading dataset {dataset_name}")
    dataset, splits = get_dataset(dataset_name=dataset_name, dataset_root=dataset_dir)
    logger.info("Dataset loaded and configuration completed")
    if dataset_name in REGRESSION_DATASETS:
        task_type = TaskType.REGRESSION
    elif dataset in CLASSIFICATION_DATASETS:
        task_type = TaskType.CLASSIFICATION
    else:
        task_type = TaskType.LINK_PREDICTION
    return activations_root, dataset, figures_dir, predictions_dir, cka_dir, task_type, splits


def get_dataset_name(cfg) -> str:
    if isinstance(cfg.dataset, str):
        dataset_name = cfg.dataset
    else:
        dataset_name = cfg.dataset.name
    return dataset_name


def get_directories(cfg: DictConfig, activations_root: Optional[Union[str, Path]], make_directories: bool) \
        -> Tuple[Union[str, Path], Path, Path, Path, Path]:
    """
    convert the directories to represent the directory in the hydra runtime environment and create then if needed
    :param cfg: project configuration
    :param activations_root: path to store the activations
    :param make_directories: whether to create the directories or not
    :return: tuple of activations, dataset, figures, predictions, and cka_dir directories
    """
    figures_dir = Path(os.getcwd(), "figures")
    predictions_dir = Path(os.getcwd(), "predictions")
    cka_dir = Path(os.getcwd(), "cka")
    dataset_dir = Path(Path(__file__).parent.resolve().parent.parent,
                       cfg.data_root)  # get path relative to the project dir
    if activations_root is None:
        activations_root = os.getcwd()
    if make_directories:
        os.makedirs(figures_dir)
        os.makedirs(predictions_dir)
        os.makedirs(cka_dir)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
    return activations_root, dataset_dir, figures_dir, predictions_dir, cka_dir


def fix_seeds(seed: int):
    """
    set the random seed and enable deterministic mode for PyTorch
    :param seed: random seed to use
    """
    pl.seed_everything(seed)
    # todo: fix the problem with warn only and remove it
    torch.use_deterministic_algorithms(True,
                                       warn_only=True)  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
