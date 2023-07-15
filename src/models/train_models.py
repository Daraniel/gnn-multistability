# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/training/node.py
#  and https://github.com/mklabunde/gnn-prediction-instability/blob/main/setup.py

import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, Iterator, Union, Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from models.gnn_models import get_model

log = logging.getLogger(__name__)


def train_models(cfg, activations_root, predictions_dir, dataset):
    """
    trains models with the given configuration on the given dataset and save their results to the predictions_dir
    :param cfg: project configuration
    :param activations_root: path to save the activations
    :param predictions_dir: path to save the predictions
    :param dataset: dataset to train on
    """
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
            model, eval_results = train_graph_classifier_model(cfg, dataset['train'], dataset['valid'], dataset['test'],
                                                               init_seed=init_seed, train_seed=seed)
        else:
            log.info(f"Training model {i + 1} out of {cfg.n_repeat} with seed {current_seed}.")
            model, eval_results = train_graph_classifier_model(cfg, dataset['train'], dataset['valid'], dataset['test'],
                                                               init_seed=init_seed, train_seed=current_seed)
        evals.append(eval_results)

        # After training, save the activations of a model
        save_dir = os.path.join(activations_root, str(current_seed))

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # os.makedirs(save_dir, parents=True, exist_ok=False)
        # if cfg.cka.use_masks:  # no need to save activations if they are not used later
        log.info("Saving model activations to %s", save_dir)
        with torch.no_grad():
            model.eval()
            act = model.activations(dataset['test'])
            for key, acts in act.items():
                save_path = os.path.join(save_dir, key + ".pt")
                torch.save(acts, save_path)
            # dataset['test'].x.to(torch.device("cpu"))

        log.info("Saving predictions")
        with torch.no_grad():
            # for dataset_type in ['train', 'valid', 'test']: # todo: include other datasets
            for dataset_type in ['test']:
                temp_preds = None
                temp_outputs = None
                dataloader = DataLoader(dataset[dataset_type], batch_size=dataset['test'].__len__(), shuffle=False)
                # noinspection PyTypeChecker
                for data in dataloader:
                    data = data.to(next(model.parameters()).device)
                    output = model(data)
                    preds = output.argmax(dim=-1)
                    outputs_test.append(
                        F.softmax(output, dim=-1).cpu().detach()
                    )
                    predictions.append(preds.cpu().detach())
                    logits_test.append(output.cpu().detach())
                    break  # TODO: update to support dataloader that have more than one batch (are not full batch)

        # Backup the trained weights currently in the working directory as checkpoint.pt
        checkpoint_dir = os.path.join(os.getcwd(), str(current_seed))
        os.makedirs(checkpoint_dir, exist_ok=True)
        if Path(os.getcwd(), "checkpoint.pt").exists():
            shutil.move(
                Path(os.getcwd(), "checkpoint.pt"),
                Path(checkpoint_dir, "checkpoint.pt"),
            )

    # Some logging and simple heuristic to catch models that are far from optimally trained
    suboptimal_models = find_suboptimal_models(evals)
    with open(Path(predictions_dir, "suboptimal_models.pkl"), "wb") as f:
        pickle.dump(suboptimal_models, f)

    with open(Path(predictions_dir, "evals.json"), "w") as f:
        json.dump(evals, f)
    with open(Path(predictions_dir, "logits_test.json"), "wb") as f:
        pickle.dump(logits_test, f)
    with open(Path(predictions_dir, "outputs_test.json"), "wb") as f:
        pickle.dump(outputs_test, f)
    with open(Path(predictions_dir, "predictions.json"), "wb") as f:
        pickle.dump(predictions, f)

    log.info("Finished training.")


def train_graph_classifier_model(cfg: DictConfig, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset,
                                 init_seed: int, train_seed: int) -> Tuple[torch.nn.Module, Dict[str, float]]:
    # Seeds are set later for training and initialization individually
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    if isinstance(cfg.cuda, str):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Build model
    log.info(f"Using model: {cfg.model.name}")
    log.info(f"Initializing model with seed={init_seed}")
    pl.seed_everything(init_seed)

    model = get_model(cfg=cfg, in_dim=train_dataset.num_features, out_dim=train_dataset.num_classes)

    log.info(f"Model has {count_parameters(model)} parameters ({count_parameters(model, trainable=True)} trainable).")

    # Set up training
    pl.seed_everything(train_seed)
    optimizer = get_optimizer(model.parameters(), cfg.optim)
    early_stopper = EarlyStopping(
        cfg.patience,
        verbose=True,
        path=Path(os.getcwd(), "checkpoint.pt"),
        trace_func=log.debug,
    )
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = cfg.n_epochs

    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_dataset.__len__(), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

    start = time.perf_counter()
    for e in range(n_epochs):
        train_loss = train_model_once(model, train_dataloader, optimizer, criterion)
        eval_results = evaluate(model, train_dataloader, valid_dataloader, test_dataloader, criterion=criterion)
        if 'train_acc' in eval_results.keys():
            log.info(
                f"time={time.perf_counter() - start:.2f} epoch={e}: "
                f"{train_loss=:.3f}, train_acc={eval_results['train_acc']:.2f}, "
                f"valid_loss={eval_results['valid_loss']:.3f}, valid_acc={eval_results['valid_acc']:.2f}"
            )
        else:
            log.info(
                f"time={time.perf_counter() - start:.2f} epoch={e}: {train_loss=:.3f}, "
                f"valid_loss={eval_results['valid_loss']:.3f}"
            )
        early_stopper(eval_results["valid_loss"], model)
        if early_stopper.early_stop and cfg.early_stopping:
            log.info(
                "Stopping training early because validation loss has not decreased"
                " after %i epochs",
                early_stopper.patience,
            )
            break

    log.info("Reverting to model with best val loss")
    if Path(early_stopper.path).exists():
        model.load_state_dict(torch.load(early_stopper.path))
    eval_results = evaluate(model, train_dataloader, valid_dataloader, test_dataloader, criterion=criterion)
    if 'train_acc' in eval_results.keys():
        log.info(
            f"train_loss={eval_results['train_loss']:.3f}, train_acc={eval_results['train_acc']:.2f}, "
            f"valid_loss={eval_results['valid_loss']:.3f}, valid_acc={eval_results['valid_acc']:.2f}, "
            f"test_loss={eval_results['test_loss']:.3f}, test_acc={eval_results['test_acc']:.2f}"
        )
    else:
        log.info(
            f"train_loss={eval_results['train_loss']:.3f}, "
            f"valid_loss={eval_results['valid_loss']:.3f}, "
            f"test_loss={eval_results['test_loss']:.3f}"
        )

    return model, eval_results


def train_model_once(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                     criterion: torch.nn.Module):
    model.train()
    total_loss = 0
    # noinspection PyTypeChecker
    for data in train_loader:
        data = data.to(next(model.parameters()).device)
        optimizer.zero_grad()
        out = model(data)
        if out.shape == data.y.shape:
            loss = criterion(out, data.y)
        else:
            loss = criterion(out, data.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def find_suboptimal_models(evals: List[Dict[str, float]], allowed_deviation: int = 2) \
        -> Dict[str, List[Tuple[int, float]]]:
    results = {}
    if 'train_acc' in evals[0].keys():
        metric = "acc"
    else:
        metric = "loss"
    for split in ["train", "valid", "test"]:
        split_results = [r[f"{split}_{metric}"] for r in evals]
        log.info(
            f"Mean %s {metric}=%.3f, Std=%.3f",
            split,
            np.mean(split_results),
            np.std(split_results),
        )
        suspicious_models: List[Tuple[int, float]] = []
        for i, acc in enumerate(split_results):
            if np.abs(acc - np.mean(split_results)) > allowed_deviation * np.std(
                    split_results
            ):
                suspicious_models.append((i, acc))
        log.info(
            f"Suspicious models (large deviation from mean {metric} on %s): %s",
            split,
            str(suspicious_models),
        )
        results[split] = suspicious_models
    return results


@torch.no_grad()
def evaluate(model: torch.nn.Module, train_dataloader: Optional[DataLoader] = None,
             valid_dataloader: Optional[DataLoader] = None, test_dataloader: Optional[DataLoader] = None,
             criterion: Optional[torch.nn.Module] = None, ) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results = {}
    for key, dataloader in zip(["train", "valid", "test"], [train_dataloader, valid_dataloader, test_dataloader]):
        if dataloader is not None:
            for data in dataloader:
                data = data.to(device)
                out = model(data)
                y_pred = out.argmax(dim=-1, keepdim=True)
                if out.shape != data.y.shape:
                    results[f"{key}_acc"] = accuracy(y_pred.view(-1), data.y)
                if criterion is not None:
                    loss = criterion(out, data.y).item()
                    results[f"{key}_loss"] = loss
                break  # TODO: update to support dataloader that have more than one batch (are not full batch)
    return results


# based on https://pytorch-geometric.readthedocs.io/en/1.7.2/_modules/torch_geometric/utils/metric.html
def accuracy(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: float
    """
    return (pred == target).sum().item() / target.numel()


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Credit: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path: Union[str, Path] = "checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, valid_loss, model):

        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_loss, model)
            self.counter = 0

    def save_checkpoint(self, valid_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.valid_loss_min = valid_loss


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/training/utils.py
def count_parameters(m: torch.nn.Module, trainable: bool = True) -> int:
    """Count the number of (trainable) parameters of a model

    Args:
        m (torch.nn.Module): model to count parameters of
        trainable (bool, optional): Whether to only count trainable parameters. Defaults to True.

    Returns:
        int: number of parameters
    """
    if trainable:
        return sum(w.numel() for w in m.parameters() if w.requires_grad)
    else:
        return sum(w.numel() for w in m.parameters())


# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/training/utils.py
def get_optimizer(params: Iterator[torch.nn.Parameter], cfg: DictConfig) -> torch.optim.Optimizer:
    """Get an optimizer as configured

    Args:
        params (Iterator[Parameter]): model.parameters()
        cfg (DictConfig): config of optimizer

    Raises:
        NotImplementedError: if trying to use optimizer that is not Adam

    Returns:
        torch.optim.Optimizer: configured optimizer
    """
    if cfg.name == "Adam":
        return torch.optim.Adam(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif cfg.name == "SGD":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError()
