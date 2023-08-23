import argparse
import logging
import os
import sys

import yaml
from omegaconf import DictConfig

from common.utils import setup_project
from evaluation.evaluate_models import evaluate_models

log = logging.getLogger(__name__)


def main(instance_path: str, start_index: int, end_index: int):
    """
    evaluate multiple instances of the trained models
    :param instance_path: path to the hydra instances of the trained model
    :param start_index: start index of the instances
    :param end_index: end index of the instances
    """
    os.chdir(instance_path)
    for i in range(start_index, end_index):
        os.chdir(f'{i}')
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
        os.chdir('..')
    log.info("Process finished successfully")


if __name__ == "__main__":
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False, default='multirun/2023-07-30/16-56-51',
                        help='Path to the trained instance, can be either relative to the project dir or absolute')

    parser.add_argument('--start', required=False, default=0, type=int, help='start index')
    parser.add_argument('--end', required=False, default=1, type=int, help='end index')
    args = parser.parse_args()

    log.setLevel(logging.INFO)
    console_handle = logging.StreamHandler(sys.stdout)
    log.addHandler(console_handle)
    main(args.path, args.start, args.end)
