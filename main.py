import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from data_loaders.get_dataset import get_dataset
from models.gnn_models import get_model_class
from models.train_models import train_graph_classifier

log = logging.getLogger(__name__)


# def train_model(model_class: torch.nn.Module.__class__, dataset: Dict[str, Dataset], cfg: DictConfig):
#     model = model_class(in_dim=dataset['train'].num_features,
#                         num_layers=cfg.model.n_layers,
#                         hidden_dim=cfg.model.hidden_dim,
#                         out_dim=dataset['train'].num_classes)
#     train_loader = DataLoader(dataset['train'], batch_size=dataset['train'].__len__(), shuffle=False)
#     val_loader = DataLoader(dataset['valid'], batch_size=dataset['valid'].__len__(), shuffle=False)
#     test_loader = DataLoader(dataset['test'], batch_size=dataset['test'].__len__(), shuffle=False)


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/setup.py
def main(cfg: DictConfig, activations_root: Optional[Union[str, Path]] = None):
    log.info("Configuring project")
    print(OmegaConf.to_yaml(cfg))

    figures_dir = Path(os.getcwd(), "figures")
    os.makedirs(figures_dir)
    predictions_dir = Path(os.getcwd(), "predictions")
    os.makedirs(predictions_dir)
    cka_dir = Path(os.getcwd(), "cka")
    os.makedirs(cka_dir)
    dataset_dir = Path(get_original_cwd(), cfg.data_root)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if activations_root is None:
        activations_root = os.getcwd()

    fix_seeds(cfg.datasplit_seed)

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

    log.info("Loading dataset")
    dataset = get_dataset(dataset_name=cfg.dataset.name, dataset_root=dataset_dir)

    log.info("Loading model")
    model_class = get_model_class(cfg.model.name)

    log.info("Training model")

    predictions: List[torch.Tensor] = []
    outputs_test: List[torch.Tensor] = []
    logits_test: List[torch.Tensor] = []
    evals: List[Dict[str, float]] = []
    seed: int = cfg.seed
    for i in range(cfg.n_repeat):
        current_seed = seed + i
        init_seed = current_seed if not cfg.keep_init_seed_constant else seed
        if cfg.keep_train_seed_constant:
            log.info(f"Training model {i + 1} out of {cfg.n_repeat} with seed {seed} (init_seed={init_seed}).")
            model, eval_results = train_graph_classifier(cfg, dataset['train'], dataset['valid'], dataset['test'],
                                                         model_class, init_seed=init_seed, train_seed=seed)
        else:
            log.info(f"Training model {i + 1} out of {cfg.n_repeat} with seed {current_seed}.")
            model, eval_results = train_graph_classifier(cfg, dataset['train'], dataset['valid'], dataset['test'],
                                                         model_class, init_seed=init_seed, train_seed=current_seed)
        evals.append(eval_results)

        # After training, save the activations of a model
        save_dir = os.path.join(activations_root, str(current_seed))
        os.makedirs(save_dir, exist_ok=False)
        if cfg.cka.use_masks:  # no need to save activations if they are not used later
            log.info("Saving model activations to %s", save_dir)
            with torch.no_grad():
                model.eval()
                assert callable(model.activations)
                act = model.activations(dataset['test'])
                for key, acts in act.items():
                    save_path = os.path.join(save_dir, key + ".pt")
                    torch.save(acts, save_path)
        log.info("Done!")

        log.info("Saving predictions")
        with torch.no_grad():
            output = model(dataset['test'])
            preds = output.argmax(dim=-1)
            outputs_test.append(
                F.softmax(output, dim=-1).cpu().detach()
            )
            predictions.append(preds.cpu().detach())
            logits_test.append(output.cpu().detach())

        # Backup the trained weights currently in the working directory as checkpoint.pt
        checkpoint_dir = os.path.join(os.getcwd(), str(current_seed))
        os.makedirs(checkpoint_dir, exist_ok=True)
        if Path(os.getcwd(), "checkpoint.pt").exists():
            shutil.move(
                Path(os.getcwd(), "checkpoint.pt"),
                Path(checkpoint_dir, "checkpoint.pt"),
            )

    log.info("Finished training.")


def fix_seeds(seed):
    pl.seed_everything(seed)
    # torch.use_deterministic_algorithms(True)  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


@hydra.main(config_path="config", config_name="main", version_base="1.2", )
def run(cfg):
    if not cfg.store_activations:
        with tempfile.TemporaryDirectory() as tmpdir:
            main(cfg, activations_root=tmpdir)
    else:
        main(cfg, activations_root=None)


if __name__ == "__main__":
    run()
