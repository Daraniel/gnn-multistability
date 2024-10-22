# based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/src/training/node.py
#  and https://github.com/mklabunde/gnn-prediction-instability/blob/main/setup.py

import json
import logging
import math
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
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from common.utils import TaskType, get_dataset_name
from data_loaders.tudataset_data_loader import SINGLE_VALUE_REGRESSION_DATASETS
from models.gnn_models import get_model, LinkPredictor

log = logging.getLogger(__name__)


def train_models(cfg, activations_root, predictions_dir, dataset: Union[Dict[str, Dataset], Dataset],
                 task_type: TaskType,
                 split_edge: Union[None, Dict[str, Tensor]]):
    """
    trains models with the given configuration on the given dataset and save their results to the predictions_dir
    :param cfg: project configuration
    :param activations_root: path to save the activations
    :param predictions_dir: path to save the predictions
    :param dataset: dataset to train on
    :param task_type: type of task (e.g., regression, classification)
    :param split_edge: split indices of dataset (if available)
    """
    log.info("Training model")

    predictions: List[torch.Tensor] = []
    outputs_test: List[torch.Tensor] = []
    logits_test: List[torch.Tensor] = []
    evals: List[Dict[str, float]] = []

    neg_predictions: List[torch.Tensor] = []
    neg_outputs_test: List[torch.Tensor] = []
    neg_logits_test: List[torch.Tensor] = []
    seed: int = cfg.seed
    main_dataset = dataset  # preserve keep raw dataset
    for i in range(cfg.n_repeat):
        current_seed = seed + i
        init_seed = current_seed if not cfg.keep_init_seed_constant else seed
        if task_type == TaskType.LINK_PREDICTION:
            dataset = get_ogbl_data(main_dataset, cfg)
        if cfg.keep_train_seed_constant:
            log.info(f"Training model {i + 1} out of {cfg.n_repeat} with seed {seed} (init_seed={init_seed}).")
            model, eval_results, predictor = train_graph_classifier_model(cfg, dataset, init_seed=init_seed,
                                                                          train_seed=seed, task_type=task_type,
                                                                          split_edge=split_edge)
        else:
            log.info(f"Training model {i + 1} out of {cfg.n_repeat} with seed {current_seed}.")
            model, eval_results, predictor = train_graph_classifier_model(cfg, dataset, init_seed=init_seed,
                                                                          train_seed=current_seed, task_type=task_type,
                                                                          split_edge=split_edge)
        evals.append(eval_results)

        # After training, save the activations of a model
        save_dir = os.path.join(activations_root, str(current_seed))

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # os.makedirs(save_dir, parents=True, exist_ok=False)
        # if cfg.cka.use_masks:  # no need to save activations if they are not used later
        log.info("Saving model activations to %s", save_dir)
        with torch.no_grad():
            model.eval()
            if task_type == TaskType.LINK_PREDICTION:
                act = model.activations(dataset)
            else:
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
                if task_type == TaskType.LINK_PREDICTION:
                    h = model(dataset)
                    pos_test_edge = split_edge['test']['edge'].to(h.device)
                    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

                    # noinspection PyTypeChecker
                    pos_test_preds = []
                    # noinspection PyTypeChecker
                    for perm in DataLoader(range(pos_test_edge.size(0)), pos_test_edge.size(0)):
                        edge = pos_test_edge[perm].t()
                        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
                    pos_test_pred = torch.cat(pos_test_preds, dim=0)
                    # preds = pos_test_pred.argmax(dim=-1)
                    preds = pos_test_pred
                    outputs_test.append(pos_test_pred.cpu().detach().view(-1, 1))
                    predictions.append(preds.cpu().detach().view(-1, 1))
                    logits_test.append(pos_test_pred.cpu().detach().view(-1, 1))

                    neg_test_preds = []
                    # noinspection PyTypeChecker
                    for perm in DataLoader(range(neg_test_edge.size(0)), neg_test_edge.size(0)):
                        edge = neg_test_edge[perm].t()
                        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
                    neg_test_pred = torch.cat(neg_test_preds, dim=0)
                    nge_preds = neg_test_pred.argmax(dim=-1)
                    neg_outputs_test.append(neg_test_pred.cpu().detach().view(-1, 1))
                    neg_predictions.append(nge_preds.cpu().detach().view(-1, 1))
                    neg_logits_test.append(neg_test_pred.cpu().detach().view(-1, 1))

                else:
                    dataloader = DataLoader(dataset[dataset_type], batch_size=dataset['test'].__len__(), shuffle=False)
                    # noinspection PyTypeChecker
                    for data in dataloader:
                        data = data.to(next(model.parameters()).device)
                        output = model(data)
                        preds = output.argmax(dim=-1)
                        if task_type == TaskType.REGRESSION:
                            outputs_test.append(output.cpu().detach())
                        else:
                            outputs_test.append(F.softmax(output, dim=-1).cpu().detach())
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
        if Path(os.getcwd(), "checkpoint_predictor.pt").exists():
            shutil.move(
                Path(os.getcwd(), "checkpoint_predictor.pt"),
                Path(checkpoint_dir, "checkpoint_predictor.ptt"),
            )

    # Some logging and simple heuristic to catch models that are far from optimally trained
    suboptimal_models = find_suboptimal_models(evals, task_type=task_type)
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
    if task_type == TaskType.LINK_PREDICTION:
        with open(Path(predictions_dir, "neg_logits_test.json"), "wb") as f:
            pickle.dump(neg_logits_test, f)
        with open(Path(predictions_dir, "neg_outputs_test.json"), "wb") as f:
            pickle.dump(neg_outputs_test, f)
        with open(Path(predictions_dir, "neg_predictions.json"), "wb") as f:
            pickle.dump(neg_predictions, f)

    log.info("Finished training.")


def train_graph_classifier_model(cfg: DictConfig, dataset: Union[Tensor, torch.nn.Embedding, Dict[str, Dataset]],
                                 init_seed: int, train_seed: int, task_type: TaskType,
                                 split_edge: Union[None, Dict[str, Tensor]]) \
        -> Tuple[torch.nn.Module, Dict[str, float], Union[None, torch.nn.Module]]:
    # Seeds are set later for training and initialization individually
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = get_device(cfg)
    log.info(f"Using device: {device}")

    # Build model
    log.info(f"Using model: {cfg.model.name}")
    log.info(f"Initializing model with seed={init_seed}")
    pl.seed_everything(init_seed)

    predictor = None
    if task_type == TaskType.LINK_PREDICTION:
        input_shape = dataset.num_features
        model = get_model(cfg=cfg, in_dim=input_shape, out_dim=cfg.model.hidden_dim, task_type=task_type)
        predictor = LinkPredictor(cfg.model.hidden_dim, cfg.model.hidden_dim, 1,
                                  cfg.model.num_layers, cfg.model.dropout_p).to(device)
        if hasattr(dataset, 'emb'):
            torch.nn.init.xavier_uniform_(dataset.emb.weight)
            # noinspection PyTypeChecker
            optimizer = get_optimizer(
                list(model.parameters()) + list(dataset.emb.parameters()) + list(predictor.parameters()), cfg.optim)
        else:
            # noinspection PyTypeChecker
            optimizer = get_optimizer(list(model.parameters()) + list(predictor.parameters()), cfg.optim)
    else:
        train_dataset = dataset['train']
        valid_dataset = dataset['valid']
        test_dataset = dataset['test']

        output_shape = train_dataset.num_classes
        if train_dataset.name in SINGLE_VALUE_REGRESSION_DATASETS:
            output_shape = 1  # HINT: use one output for single value regression datasets
        input_shape = train_dataset.num_features

        # use batch training for regression dataset since it's too big but full batch for other datasets that are small
        if task_type == TaskType.REGRESSION:
            # train_dataloader = DataLoader(train_dataset, batch_size=math.ceil(train_dataset.__len__() / 32),
            #                               shuffle=False)
            # valid_dataloader = DataLoader(valid_dataset, batch_size=math.ceil(valid_dataset.__len__() / 32),
            #                               shuffle=False)

            train_dataloader = DataLoader(train_dataset[0], batch_size=2,
                                          shuffle=False)
            valid_dataloader = DataLoader(valid_dataset[0], batch_size=2,
                                          shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False)
            valid_dataloader = DataLoader(valid_dataset, batch_size=valid_dataset.__len__(), shuffle=False)
        # test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
        test_dataloader = DataLoader(test_dataset[0], batch_size=2, shuffle=False)
        model = get_model(cfg=cfg, in_dim=input_shape, out_dim=output_shape, task_type=task_type)
        optimizer = get_optimizer(model.parameters(), cfg.optim)

    # log.info(f"Model has {count_parameters(model)} parameters ({count_parameters(model, trainable=True)} trainable).")

    # Set up training
    pl.seed_everything(train_seed)
    early_stopper = EarlyStopping(
        cfg.patience,
        verbose=True,
        path=Path(os.getcwd(), "checkpoint.pt"),
        trace_func=log.debug,
    )
    if task_type == TaskType.REGRESSION:
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    n_epochs = cfg.n_epochs

    model.to(device)

    start = time.perf_counter()
    for e in range(n_epochs):
        if task_type == TaskType.LINK_PREDICTION:
            # noinspection PyUnboundLocalVariable
            train_loss = train_link_prediction_model_once(model, predictor, dataset, split_edge, optimizer, cfg)
            # noinspection PyUnboundLocalVariable
            eval_results = evaluate_link_prediction(model, predictor, dataset, split_edge, cfg)
        else:
            # noinspection PyUnboundLocalVariable
            train_loss = train_model_once(model, train_dataloader, optimizer, criterion)
            # noinspection PyUnboundLocalVariable
            eval_results = evaluate(model, task_type, train_dataloader, valid_dataloader, test_dataloader,
                                    criterion=criterion)
        if e == 0:
            # HINT: print number of parameters after handling first input to prevent some errors
            log.info(f"Model has {count_parameters(model)} parameters"
                     f" ({count_parameters(model, trainable=True)} trainable).")
            if task_type == TaskType.LINK_PREDICTION:
                log.info(f"Link Predictor has {count_parameters(predictor)} parameters"
                         f" ({count_parameters(predictor, trainable=True)} trainable).")
                if hasattr(dataset, 'emb'):
                    log.info(f"Embedings has {count_parameters(dataset.emb)} parameters"
                             f" ({count_parameters(dataset.emb, trainable=True)} trainable).")
        if task_type == TaskType.CLASSIFICATION:
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
        early_stopper(eval_results["valid_loss"], model, predictor)
        if early_stopper.early_stop and cfg.early_stopping:
            log.info(
                "Stopping training early because validation loss has not decreased"
                " after %i epochs",
                early_stopper.patience,
            )
            break

    log.info("Reverting to model with best val loss")
    return
    if Path(early_stopper.path).exists():
        model.load_state_dict(torch.load(early_stopper.path))
        if predictor is not None:
            predictor.load_state_dict(torch.load(early_stopper.path_predictor))
    if task_type == TaskType.LINK_PREDICTION:
        # noinspection PyUnboundLocalVariable
        eval_results = evaluate_link_prediction(model, predictor, dataset, split_edge, cfg)
    else:
        # noinspection PyUnboundLocalVariable
        eval_results = evaluate(model, task_type, train_dataloader, valid_dataloader, test_dataloader,
                                criterion=criterion)
    if task_type == TaskType.CLASSIFICATION:
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

    return model, eval_results, predictor


# this function is based on https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
def train_link_prediction_model_once(model: torch.nn.Module, predictor: torch.nn.Module, data,
                                     splits: Dict[str, Tensor], optimizer: torch.optim.Optimizer, cfg: DictConfig):
    device = next(model.parameters()).device
    pos_train_edge = splits['train']['edge'].to(device)
    model.train()
    predictor.train()

    # TODO: remove
    # if isinstance(data.x, torch.nn.Embedding):
    #     torch.nn.init.xavier_uniform_(data.x.weight)

    total_loss = total_examples = 0
    # noinspection PyTypeChecker
    for perm in DataLoader(range(pos_train_edge.size(0)), pos_train_edge.size(0), shuffle=True):
        optimizer.zero_grad()

        h = model(data)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def train_model_once(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                     criterion: torch.nn.Module):
    model.train()
    total_loss = 0
    # noinspection PyTypeChecker
    # for data in train_loader:
    data = train_loader.dataset
    data = data.to(next(model.parameters()).device)
    optimizer.zero_grad()
    out = model(data)
    if out.shape == data.y.shape:
        loss = criterion(out, data.y)
    elif out.shape[0] == data.y.shape[0] and out.shape[1] == 1 and len(data.y.shape) == 1 and len(out.shape) == 2:
        # HINT: y is flattened but not output
        loss = criterion(out, data.y.view(out.shape))
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


def find_suboptimal_models(evals: List[Dict[str, float]], task_type: TaskType, allowed_deviation: int = 2) \
        -> Dict[str, List[Tuple[int, float]]]:
    results = {}
    if task_type != TaskType.REGRESSION:
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


# this function is based on https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
@torch.no_grad()
def evaluate_link_prediction(model: torch.nn.Module, predictor: torch.nn.Module,
                             data: Union[Tensor, torch.nn.Embedding], split_edge: Dict[str, Tensor], cfg: DictConfig) \
        -> Dict[str, float]:
    model.eval()
    predictor.eval()
    h = model(data)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    # noinspection PyTypeChecker
    for perm in DataLoader(range(pos_train_edge.size(0)), pos_train_edge.size(0)):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    pos_train_pred_loss = -torch.log(pos_train_pred + 1e-15).mean()

    pos_valid_preds = []
    # noinspection PyTypeChecker
    for perm in DataLoader(range(pos_valid_edge.size(0)), pos_valid_edge.size(0)):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    pos_valid_pred_loss = -torch.log(pos_valid_pred + 1e-15).mean()

    neg_valid_preds = []
    # noinspection PyTypeChecker
    for perm in DataLoader(range(neg_valid_edge.size(0)), neg_valid_edge.size(0)):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    neg_valid_pred_loss = -torch.log(1 - neg_valid_pred + 1e-15).mean()

    h = model(data)

    pos_test_preds = []
    # noinspection PyTypeChecker
    for perm in DataLoader(range(pos_test_edge.size(0)), pos_test_edge.size(0)):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    pos_test_pred_loss = -torch.log(pos_test_pred + 1e-15).mean()

    neg_test_preds = []
    # noinspection PyTypeChecker
    for perm in DataLoader(range(neg_test_edge.size(0)), neg_test_edge.size(0)):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    neg_test_pred_loss = -torch.log(1 - neg_test_pred + 1e-15).mean()

    results = {}
    evaluator = Evaluator(name=f'ogbl-{get_dataset_name(cfg).lower()}')
    # for K in [10, 50, 100]:
    #     evaluator.K = K
    train_hits = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })

    for key, value in train_hits.items():
        results[f'train_{key}'] = value
    for key, value in valid_hits.items():
        results[f'valid_{key}'] = value
    for key, value in test_hits.items():
        results[f'test_{key}'] = value

    results['train_loss'] = pos_train_pred_loss.item()
    results['valid_loss'] = pos_valid_pred_loss.item() + neg_valid_pred_loss.item()
    results['test_loss'] = pos_test_pred_loss.item() + neg_test_pred_loss.item()
    return results


def get_ogbl_data(dataset, cfg: DictConfig) -> Union[Tensor, torch.nn.Embedding]:
    device = get_device(cfg)
    data = dataset[0].to(device)
    dataset_name = get_dataset_name(cfg)
    if dataset_name == 'ppa':
        data.x = data.x.to(torch.float)
    elif dataset_name == 'collab':
        # data.x = data.x.to(torch.float)
        # data.adj_t = data.adj_t.to(torch.float)
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
    elif dataset_name == 'ddi':
        emb = torch.nn.Embedding(data.adj_t.size(0),
                                 cfg.model.hidden_dim)
        data.emb = emb.to(device)
    return data


def get_device(cfg: DictConfig):
    if isinstance(cfg.cuda, str):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
    return device


@torch.no_grad()
def evaluate(model: torch.nn.Module, task_type: TaskType, train_dataloader: Optional[DataLoader] = None,
             valid_dataloader: Optional[DataLoader] = None, test_dataloader: Optional[DataLoader] = None,
             criterion: Optional[torch.nn.Module] = None, ) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results = {}
    for key, dataloader in zip(["train", "valid", "test"], [train_dataloader, valid_dataloader, test_dataloader]):
        if dataloader is not None:
            outputs = []
            y_preds = []
            ys = []
            data = dataloader.dataset
            # for data in dataloader:
            data = data.to(device)
            out = model(data)
            outputs.append(out.cpu().detach())
            if task_type != TaskType.REGRESSION:
                y_pred = out.argmax(dim=-1, keepdim=True)
                y_preds.append(y_pred.cpu().detach())
            ys.append(data.y.cpu().detach())

            outputs = torch.cat(outputs)
            ys = torch.cat(ys)
            if task_type != TaskType.REGRESSION:
                y_preds = torch.cat(y_preds)
                results[f"{key}_acc"] = accuracy(y_preds.view(-1), ys)
            if criterion is not None:
                if (outputs.shape[0] == ys.shape[0] and len(outputs.shape) == 2 and outputs.shape[1] == 1
                        and len(ys.shape) == 1):
                    # HINT: y is flattened but not output
                    loss = criterion(outputs, ys.view(outputs.shape)).item()
                else:
                    loss = criterion(outputs, ys).item()
                results[f"{key}_loss"] = loss
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
        self.path_predictor = f"{Path(path).stem}_predictor.pt"
        self.trace_func = trace_func

    def __call__(self, valid_loss, model, predictor):

        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, predictor)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, predictor)
            self.counter = 0

    def save_checkpoint(self, valid_loss, model, predictor):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        if predictor is not None:
            torch.save(predictor.state_dict(), self.path_predictor)
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
