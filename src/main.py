import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import hydra
from omegaconf import DictConfig

from common.utils import setup_project
from evaluation.evaluate_models import evaluate_models
from models.train_models import train_models

log = logging.getLogger(__name__)


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/setup.py
def main(cfg: DictConfig, activations_root: Optional[Union[str, Path]] = None):
    """
    train and evaluate the models
    :param cfg: project configuration
    :param activations_root: path to store the activations
    """
    activations_root, dataset, figures_dir, predictions_dir, cka_dir, task_type = setup_project(
        cfg, activations_root, log)
    train_models(cfg, activations_root, predictions_dir, dataset, task_type=task_type)
    evaluate_models(cfg, activations_root, dataset, figures_dir, predictions_dir, cka_dir, task_type=task_type)
    log.info("Process finished successfully")


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def run(cfg=None):
    if not cfg.store_activations:
        with tempfile.TemporaryDirectory() as tmpdir:
            main(cfg, activations_root=tmpdir)
    else:
        main(cfg, activations_root=None)


if __name__ == "__main__":
    os.chdir('..')
    run()
