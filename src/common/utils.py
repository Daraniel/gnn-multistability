import logging
import os
from pathlib import Path
from typing import Union, Any, Dict, Tuple, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from torch_geometric.data import Dataset

from data_loaders.get_dataset import get_dataset


def setup_project(cfg: DictConfig, activations_root: Optional[Union[str, Path]], logger: logging.Logger,
                  make_directories: bool = True) -> Tuple[Union[str, Any], Dict[str, Dataset], Path, Path]:
    """
    set up the project and handle the directories
    :param cfg: project configuration
    :param activations_root: path to store the activations
    :param logger: project logger
    :param make_directories: whether to create the directories or not
    :return: tuple of activations, dataset, figures, and predictions directories
    """
    logger.info("Configuring project")
    print(OmegaConf.to_yaml(cfg))
    activations_root, dataset_dir, figures_dir, predictions_dir = get_directories(cfg, activations_root,
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
    logger.info("Loading dataset")
    dataset = get_dataset(dataset_name=cfg.dataset.name, dataset_root=dataset_dir)
    logger.info("Dataset loaded and configuration completed")
    return activations_root, dataset, figures_dir, predictions_dir


def get_directories(cfg: DictConfig, activations_root: Optional[Union[str, Path]], make_directories: bool) \
        -> Tuple[Union[str, Path], Path, Path, Path]:
    """
    convert the directories to represent the directory in the hydra runtime environment and create then if needed
    :param cfg: project configuration
    :param activations_root: path to store the activations
    :param make_directories: whether to create the directories or not
    :return: tuple of activations, dataset, figures, and predictions directories
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
    return activations_root, dataset_dir, figures_dir, predictions_dir


def fix_seeds(seed: int):
    """
    set the random seed and enable deterministic mode for PyTorch
    :param seed: random seed to use
    """
    pl.seed_everything(seed)
    torch.use_deterministic_algorithms(True)  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
