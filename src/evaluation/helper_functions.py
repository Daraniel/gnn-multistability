import itertools
import math
import os
from pathlib import Path
from typing import List, Union, Callable

import numpy as np
from tqdm import tqdm

from evaluation import experiments


# todo: cleanup
def pairwise_apply_function_full(dirnames: List[str], cka_dir: Union[str, Path], split_name: str,  # idx: np.ndarray,
                                 activations_root: Path, function_to_use: Callable, calculating_function_name: str,
                                 save_to_disk: bool = True) -> List[np.ndarray]:
    """Compute full CKA matrices for pairs of models (and saves them to disk)

    Args:
        dirnames (List[str]): List of directories containing activations of different models
        idx (np.ndarray): indexes the activations so only a subset is used
        cka_dir (Union[str, Path]): path to a directory where results are saved
        split_name (str): identifier for used subset (idx)
        save_to_disk (bool): Whether to save cka matrices to disk. Defaults to True.
    """
    cka_matrices = []
    pair_length: int = 2
    for seed_pair in itertools.combinations(sorted(dirnames), pair_length):
        # 1. Extract all filenames related to saved activations and sort them so
        # plots using them have a fixed structure
        fnames = experiments.find_activation_fnames(seed_pair, activations_root)

        # 2. Für alle Kombinationen, berechne die CKA Werte und speichere sie ab
        # Wir indexen die raw activations mit dem data_split für detailierte Ergebnisse
        cka_values = np.zeros((len(fnames[0]), len(fnames[1])))
        with tqdm(total=cka_values.size, desc=f"Computing {calculating_function_name} values") as t:
            for i, fname1 in enumerate(fnames[0]):
                for j, fname2 in enumerate(fnames[1]):
                    x = experiments.load_representation(
                        os.path.join(os.getcwd(), seed_pair[0], f"{fname1}.pt")
                        # )[idx]
                    )
                    y = experiments.load_representation(
                        os.path.join(os.getcwd(), seed_pair[1], f"{fname2}.pt")
                        # )[idx]
                    )
                    cka_values[i, j] = function_to_use(x, y)
                    t.update()
        if save_to_disk:
            np.save(
                str(Path(cka_dir, f"{calculating_function_name}_{split_name}_{'_'.join(seed_pair)}.npy")),
                cka_values,
            )
        cka_matrices.append(cka_values)
    return cka_matrices


# todo: update
def pairwise_apply_function_diag(dirnames: List[str], cka_dir: Union[str, Path], split_name: str,  # idx: np.ndarray,
                                 activations_root: Path, function_to_use: Callable, calculating_function_name: str,
                                 save_to_disk: bool = True) -> List[
    np.ndarray]:
    cka_matrices = []
    pair_length: int = 2
    with tqdm(
            desc=f"Pairwise {calculating_function_name} computation (diag)",
            total=math.comb(len(dirnames), pair_length),
    ) as t:
        for seed_pair in itertools.combinations(sorted(dirnames), pair_length):
            # 1. Extract all filenames related to saved activations and sort them so
            # plots using them have a fixed structure
            fnames = experiments.find_activation_fnames(seed_pair, activations_root)

            # 2. Only activations of corresponding layer are compared via CKA
            # Empty cells are encoded as -inf
            if len(fnames[0]) != len(fnames[1]):
                raise ValueError(
                    "Models have different depth. No clear correspondence between layers!"
                )
            cka_values = np.ones((len(fnames[0]), len(fnames[1]))) * -np.inf
            for i, (fname1, fname2) in enumerate(zip(fnames[0], fnames[1])):
                x = experiments.load_representation(
                    os.path.join(activations_root, seed_pair[0], f"{fname1}.pt")
                    # )[idx]
                )
                y = experiments.load_representation(
                    os.path.join(activations_root, seed_pair[1], f"{fname2}.pt")
                    # )[idx]
                )
                cka_values[i, i] = function_to_use(x, y)
            if save_to_disk:
                np.save(
                    str(Path(cka_dir, f"{calculating_function_name}_{split_name}_{'_'.join(seed_pair)}.npy")),
                    cka_values,
                )
            cka_matrices.append(cka_values)
            t.update()
    return cka_matrices


# todo: cleanup
def cka_matrix(dirnames: List[str], cka_dir: Union[str, Path], split_name: str, mode: str,  # idx: np.ndarray,
               save_to_disk: bool, activations_root: Path, function_to_use: Callable, calculating_function_name: str) \
        -> List[np.ndarray]:
    if mode == "full":
        ckas = pairwise_apply_function_full(
            dirnames, cka_dir, split_name, activations_root, function_to_use, calculating_function_name, save_to_disk
        )
    elif mode == "diag":
        ckas = pairwise_apply_function_diag(
            dirnames, cka_dir, split_name, activations_root, function_to_use, calculating_function_name, save_to_disk,
        )
    else:
        raise ValueError(f"Unknown {calculating_function_name} mode: {mode}")
    return ckas
