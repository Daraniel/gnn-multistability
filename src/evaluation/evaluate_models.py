import itertools
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Any, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric
from omegaconf import DictConfig
from scipy.stats import entropy

import evaluation.predictions
import plots

log = logging.getLogger(__name__)


def evaluate_models(cfg: DictConfig, dataset: Dict[str, torch_geometric.data.Dataset], figures_dir: Path,
                    predictions_dir: Path) -> None:
    """
    evaluate the specified model
    :param cfg: project configuration
    :param dataset: dataset
    :param figures_dir: location of the figures
    :param predictions_dir: location of the predictions
    """
    log.info("Evaluating models")

    with open(Path(predictions_dir, "evals.json"), "r") as f:
        evals = json.load(f)
    with open(Path(predictions_dir, "logits_test.json"), "rb") as f:
        logits_test = pickle.load(f)
    with open(Path(predictions_dir, "outputs_test.json"), "rb") as f:
        outputs_test = pickle.load(f)
    with open(Path(predictions_dir, "predictions.json"), "rb") as f:
        predictions = pickle.load(f)

    # todo: include other datasets
    classification_stability_experiments(
        cfg=cfg,
        predictions_dir=predictions_dir,
        figures_dir=figures_dir,
        predictions=predictions,
        outputs_test=outputs_test,
        logits_test=logits_test,
        test_dataset=dataset['test'],
        evals=evals,
    )

    # todo: include other datasets
    # todo: fix bug and uncomment
    # cka_experiments(
    #     cfg=cfg,
    #     figures_dir=figures_dir,
    #     predictions_dir=predictions_dir,
    #     cka_dir=cka_dir,
    #     activations_root=activations_root,
    # )

    log.info("Finished evaluating models")


# rest of the code based on https://github.com/mklabunde/gnn-prediction-instability/blob/main/setup.py
def clean_list(lst: List[Any], items_to_remove: List[Any]) -> None:
    for item in items_to_remove:
        if item in lst:
            lst.remove(item)


def save_heatmap(
        ids: Tuple[str, str],
        ticklabels: Tuple[List[float], List[float]],
        vals: np.ndarray,
        split: str = "",
) -> None:
    """Save a heatmap of CKA values

    Args:
        ids (Tuple[str, str]): Names for the two models. First one is yaxis, second one xaxis
        ticklabels (Tuple[List[float], List[float]]): (yticks, xticks)
        vals (np.ndarray): CKA scores
        split (str): which data split is plotted. Is added to filename and plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(len(ticklabels[0]) + 2, len(ticklabels[1])))
    hm = sns.heatmap(
        vals[::-1],
        yticklabels=list(reversed(list((map(str, ticklabels[0]))))),
        xticklabels=list(map(str, ticklabels[1])),
        ax=ax,
        annot=True,
        vmin=0,
        vmax=1,
    )
    plt.suptitle(f"Linear CKA between {ids[0]} and {ids[1]} ({split})")
    savepath = os.path.join(
        os.getcwd(), "figures", f"cka_{'_'.join(list(ids) + [split])}.pdf"
    )
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
        log.debug("Created %s directory", os.path.dirname(savepath))
    plt.savefig(savepath)
    plt.close()


def classification_stability_experiments(cfg: DictConfig, predictions_dir: Path,
                                         figures_dir: Path, predictions: List[torch.Tensor],
                                         outputs_test: List[torch.Tensor], logits_test: List[torch.Tensor],
                                         test_dataset: torch_geometric.data.Dataset, evals: List[Dict[str, float]]):
    log.info("Calculating stability of predictions...")
    preval_df = []
    nodewise_distr_path = Path(predictions_dir, f"nodewise_distr.npy")
    distr = evaluation.predictions.classification_node_distr(
        predictions, test_dataset.num_classes  # type:ignore
    )
    np.save(str(nodewise_distr_path), distr)
    # for split_name, idx in split_idx.items():
    for split_name in ['test']:
        # filtered_preds = [p[idx] for p in predictions]
        filtered_preds = [p for p in predictions]
        prevalences = evaluation.predictions.classification_prevalence(
            filtered_preds, test_dataset.num_classes  # type:ignore
        )
        for key, val in prevalences.items():
            preval_df.append((split_name, key, val[0], val[1]))

        frac_stable = evaluation.predictions.fraction_stable_predictions(
            filtered_preds
        )
        log.info("Predictions (%s) stable over all models: %.2f", split_name, frac_stable)
    prevalences_path = Path(predictions_dir, "prevalences.csv")
    pd.DataFrame.from_records(preval_df).to_csv(prevalences_path)
    # todo: update
    plots.save_class_prevalence_plots(
        test_dataset[0].y,  # type:ignore
        # split_idx["test"],
        prevalences_path=prevalences_path,
        savepath=figures_dir,
        dataset_name=cfg.dataset.name,
    )
    plots.save_node_instability_distribution(
        # split_idx["test"],
        prediction_distr_path=nodewise_distr_path,
        savepath=figures_dir,
        dataset_name=cfg.dataset.name,
    )

    # Compare the model output distributions to the prediction distribution
    logits_test: np.ndarray = torch.stack(logits_test, dim=0).numpy()
    probas_test: np.ndarray = torch.stack(outputs_test, dim=0).numpy()
    avg_output_entropy = np.mean(entropy(probas_test, axis=2), axis=0)
    # predictions_entropy = entropy(distr[split_idx["test"].numpy()], axis=1) # todo: update
    predictions_entropy = entropy(distr, axis=1)
    plots.node_stability.save_scatter_correlation(
        predictions_entropy,
        avg_output_entropy,
        "Prediction Entropy",
        "Average Model Output Entropy",
        "Prediction Entropy - Avg Model Output Entropy: %s",
        Path(figures_dir, "entropy_scatter.jpg"),
    )

    # Compare the output and prediction entropy to node properties
    test_dataset[0].edge_index = torch_geometric.utils.to_undirected(  # type:ignore
        test_dataset[0].edge_index  # type:ignore
    )

    # g = torch_geometric.utils.to_networkx(test_dataset[0])
    # todo: maybe update?
    # nodes = np.array([i for i, is_test in enumerate(split_idx["test"]) if is_test])
    # degrees = np.asarray([d for _, d in nx.degree(g, nbunch=nodes)])

    # todo: possible fix? I think it needs to get updated to pick the best class for each graph
    #  (like taking max of values?)
    # graphs = [torch_geometric.utils.to_networkx(test_dataset[i]) for i in range(len(test_dataset))]
    # degrees = np.asarray([nx.degree(g) for g in graphs])

    # # todo: fix bug and restore (bug is related to difference in shape of  degrees and predictions_entropy
    # plots.node_stability.save_scatter_correlation(
    #     degrees,
    #     predictions_entropy,
    #     "Degrees",
    #     "Prediction Entropy",
    #     "Degree - Prediction Entropy: %s",
    #     Path(figures_dir, "degree_predentropy.jpg"),
    # )
    # plots.node_stability.save_scatter_correlation(
    #     degrees,
    #     avg_output_entropy,
    #     "Degrees",
    #     "Avg Model Output Entropy",
    #     "Degree - Avg Output Entropy: %s",
    #     Path(figures_dir, "degree_outputentropy.jpg"),
    # )

    # Compare models pairwise w.r.t. identical predictions
    pi_distr = evaluation.predictions.pairwise_instability(
        preds=probas_test.argmax(axis=2), figurepath=figures_dir
    )
    np.save(str(Path(predictions_dir, "pi_distr.npy")), pi_distr)

    norm_pi_distr = evaluation.predictions.normalized_pairwise_instability(
        preds=probas_test.argmax(axis=2),
        accs=np.asarray([e["test_acc"] for e in evals]),
        figurepath=figures_dir,
    )
    np.save(str(Path(predictions_dir, "normpi_distr.npy")), norm_pi_distr)

    symkl_distr = evaluation.predictions.pairwise_sym_kldiv(
        outputs=logits_test, figurepath=figures_dir,
    )
    np.save(str(Path(predictions_dir, "symkl_distr.npy")), symkl_distr)

    l1_distr = evaluation.predictions.pairwise_l1loss(
        probas_test, figurepath=figures_dir
    )
    np.save(str(Path(predictions_dir, "l1_distr.npy")), l1_distr)

    (
        true_diffs,
        false_diffs,
    ) = evaluation.predictions.pairwise_conditioned_instability(
        probas_test.argmax(axis=2),
        test_dataset[0].y.cpu(),  # type:ignore
    )
    np.save(str(Path(predictions_dir, "true_pi_distr.npy")), true_diffs)
    np.save(str(Path(predictions_dir, "false_pi_distr.npy")), false_diffs)


def cka_experiments(
        cfg: DictConfig,
        figures_dir: Path,
        predictions_dir: Path,
        cka_dir: Path,
        activations_root: Path,
):
    log.info("Starting pairwise CKA computation.")
    # Jetzt startet die Analyse auf allen paaren der trainierten Modelle
    accuracy_records: List[Tuple[str, str, str, float]] = []
    # todo: update
    # for split_name, idx in split_idx.items():
    for split_name in ['test']:
        if split_name not in cfg.cka.use_masks:
            log.info("Skipping CKA analysis for %s", split_name)
            continue

        log.info("Starting CKA analysis for %s", split_name)
        # idx = idx.numpy() # todo: maybe update?
        pair_length = 2
        # Every model has its own subdirectory, but there are also other output
        # directories, which we have to remove to only iterate over pairs of model dirs
        _, dirnames, _ = next(os.walk(os.getcwd(), ))
        log.debug(f"Found dirnames: {dirnames}. Removing output directories.")
        clean_list(
            dirnames,
            items_to_remove=[
                ".hydra",
                figures_dir.parts[-1],
                cka_dir.parts[-1],
                predictions_dir.parts[-1],
            ],
        )

        cka_matrices = evaluation.cka_matrix(
            dirnames=dirnames,
            # idx=idx,
            cka_dir=cka_dir,
            split_name=split_name,
            mode=cfg.cka.mode,
            save_to_disk=cfg.cka.save_to_disk,
            activations_root=activations_root,
        )

        # ------------------------------------------------------------------------------
        log.info("Finished CKA computation. Preparing output for %s.", split_name)

        # Jetzt müssen die Ergbenisse der Paare noch aggregiert werden
        cka_matrices = np.array(cka_matrices)
        np.save(str(cka_dir / f"ckas_{split_name}.npy"), cka_matrices)
        cka_mean = np.mean(cka_matrices, axis=0)
        cka_std = np.std(cka_matrices, axis=0)
        log.debug("CKA matrices shape: %s", (cka_matrices.shape,))
        log.debug("Mean CKA shape: %s", (cka_mean.shape,))

        plots.save_cka_diagonal(
            cka_matrices, Path(figures_dir, f"cka_diag_{split_name}.pdf"),
        )

        if cfg.cka.mode == "full":
            for i, seed_pair in enumerate(
                    itertools.combinations(sorted(dirnames), pair_length)
            ):
                # Extract the activation filenames again, so we can use them as ticklabels in plots
                fnames = evaluation.find_activation_fnames(
                    seed_pair, activations_root
                )
                save_heatmap(
                    seed_pair,
                    (fnames[0], fnames[1]),
                    cka_matrices[i],
                    split=split_name,
                )
                accuracy_records.append(
                    (
                        seed_pair[0],
                        seed_pair[1],
                        split_name,
                        evaluation.accuracy_layer_identification(cka_matrices[i]),
                    )
                )
                if i == 0:
                    # To create heatmaps for mean and std CKA, we need ticklabels, which we have inside this loop
                    save_heatmap(
                        ("mean", "mean"),
                        (fnames[0], fnames[1]),
                        cka_mean,
                        split=split_name,
                    )
                    accuracy_records.append(
                        (
                            "mean",
                            "mean",
                            split_name,
                            evaluation.accuracy_layer_identification(cka_mean),
                        )
                    )
                    save_heatmap(
                        ("std", "std"),
                        (fnames[0], fnames[1]),
                        cka_std,
                        split=split_name,
                    )
                    accuracy_records.append(
                        (
                            "std",
                            "std",
                            split_name,
                            evaluation.accuracy_layer_identification(cka_std),
                        )
                    )
        pd.DataFrame.from_records(
            accuracy_records, columns=[0, 1, "split", "acc"]
        ).to_csv(os.path.join(os.getcwd(), "layer_identification.csv"))