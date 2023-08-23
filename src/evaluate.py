import argparse
import logging
import os
import sys

import yaml
from omegaconf import DictConfig

from common.utils import setup_project
from evaluation.evaluate_models import evaluate_models

log = logging.getLogger(__name__)


def main(instance_path: str):
    """
    evaluate the trained model
    :param instance_path: path to the hydra instance of the trained model
    """
    os.chdir(instance_path)
    with open('.hydra/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open('.hydra/overrides.yaml', 'r') as f:
        overrides = yaml.safe_load(f)

    # replace the values of the original config with the override values
    for item in overrides:
        key, value = item.split('=')
        cfg[key] = value
    cfg = DictConfig(cfg)

    if cfg.store_activations:
        activations_root = cfg.store_activations
    else:
        activations_root = None

    activations_root, dataset, figures_dir, predictions_dir, cka_dir, task_type = setup_project(
        cfg, activations_root, log, make_directories=False)
    evaluate_models(cfg, activations_root, dataset, figures_dir, predictions_dir, cka_dir, task_type=task_type)
    log.info("Process finished successfully")


if __name__ == "__main__":
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False, default='outputs/2023-07-12/11-30-48',
                        help='Path to the trained instance, can be either relative to the project dir or absolute')
    args = parser.parse_args()
    path = args.path

    log.setLevel(logging.INFO)
    console_handle = logging.StreamHandler(sys.stdout)
    log.addHandler(console_handle)
    main(path)
