# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/training/node.py
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Iterator, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch_geometric.utils
from omegaconf import DictConfig
from torch_geometric.data import Dataset

log = logging.getLogger(__name__)


def train_graph_classifier(cfg: DictConfig, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset,
                           model_class: torch.nn.Module.__class__, init_seed: int, train_seed: int) \
        -> Tuple[torch.nn.Module, Dict[str, float]]:
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

    model = model_class(in_dim=train_dataset.num_features,
                        num_layers=cfg.model.n_layers,
                        hidden_dim=cfg.model.hidden_dim,
                        out_dim=train_dataset.num_classes)

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
    train_dataset = train_dataset.data.to(str(device))
    valid_dataset = valid_dataset.data.to(str(device))
    test_dataset = test_dataset.data.to(str(device))

    start = time.perf_counter()
    for e in range(n_epochs):
        train_loss = train(model, train_dataset, optimizer, criterion)
        eval_results = evaluate(model, train_dataset, valid_dataset, test_dataset, criterion=criterion)
        log.info(
            f"time={time.perf_counter() - start:.2f} epoch={e}: "
            f"{train_loss=:.3f}, train_acc={eval_results['train_acc']:.2f}, "
            f"val_loss={eval_results['val_loss']:.3f}, val_acc={eval_results['val_acc']:.2f}"
        )
        early_stopper(eval_results["val_loss"], model)
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
    eval_results = evaluate(model, train_dataset, valid_dataset, test_dataset, criterion=criterion)
    log.info(
        f"train_loss={eval_results['train_loss']:.3f}, train_acc={eval_results['train_acc']:.2f}, "
        f"val_loss={eval_results['val_loss']:.3f}, val_acc={eval_results['val_acc']:.2f}, "
        f"test_loss={eval_results['test_loss']:.3f}, test_acc={eval_results['test_acc']:.2f}"
    )

    return model, eval_results


def train(model: torch.nn.Module, train_dataset: Dataset, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
    model.train()
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=5, shuffle=False)

    total_loss = 0
    for data in train_loader:
        # out = model(train_dataset)
        # loss = criterion(out, train_dataset.y.view(-1))
        # loss = F.nll_loss(out, train_dataset.y.view(-1))
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()

    # return loss.item()
    return total_loss / len(train_loader.dataset)

    # train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=50, shuffle=False)
    # for data in train_loader:
    #     out = model(data)
    #     loss = criterion(out, data.y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     return loss.item()


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, train_dataset: Optional[Dataset] = None, valid_dataset: Optional[Dataset] = None,
             test_dataset: Optional[Dataset] = None, criterion: Optional[torch.nn.Module] = None, ) -> Dict[str, float]:
    model.eval()
    results = {}
    for key, dataset in zip(["train", "valid", "test"], [train_dataset, valid_dataset, test_dataset]):
        if dataset is not None:
            out = model(dataset)
            y_pred = out.argmax(dim=-1, keepdim=True)
            results[f"{key}_acc"] = torch_geometric.utils.metric.accuracy(
                y_pred.view(-1), dataset.y
            )
            if criterion is not None:
                loss = criterion(out, dataset.y).item()
                results[f"{key}_loss"] = loss
    return results


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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
