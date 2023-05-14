"""Copied from https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb"""

import itertools
import math
import os
from pathlib import Path
from typing import List, Union

import numpy as np
from tqdm import tqdm

from evaluation import experiments


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x ** 2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y ** 2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)


# todo: update
def pairwise_cka_full(
    dirnames: List[str],
    # idx: np.ndarray,
    cka_dir: Union[str, Path],
    split_name: str,
    activations_root: Path,
    save_to_disk: bool = True,
) -> List[np.ndarray]:
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
        fnames = experiments.find_activation_fnames(
            seed_pair, activations_root
        )

        # 2. Für alle Kombinationen, berechne die CKA Werte und speichere sie ab
        # Wir indexen die raw activations mit dem data_split für detailierte Ergebnisse
        cka_values = np.zeros((len(fnames[0]), len(fnames[1])))
        with tqdm(total=cka_values.size, desc="Computing CKA values") as t:
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
                    cka_values[i, j] = feature_space_linear_cka(x, y)
                    t.update()
        if save_to_disk:
            np.save(
                str(Path(cka_dir, f"cka_{split_name}_{'_'.join(seed_pair)}.npy")),
                cka_values,
            )
        cka_matrices.append(cka_values)
    return cka_matrices


# todo: update
def pairwise_cka_diag(
    dirnames: List[str],
    # idx: np.ndarray,
    cka_dir: Union[str, Path],
    split_name: str,
    activations_root: Path,
    save_to_disk: bool = True,
) -> List[np.ndarray]:
    cka_matrices = []
    pair_length: int = 2
    with tqdm(
        desc="Pairwise CKA computation (diag)",
        total=math.comb(len(dirnames), pair_length),
    ) as t:
        for seed_pair in itertools.combinations(sorted(dirnames), pair_length):
            # 1. Extract all filenames related to saved activations and sort them so
            # plots using them have a fixed structure
            fnames = experiments.find_activation_fnames(
                seed_pair, activations_root
            )

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
                cka_values[i, i] = feature_space_linear_cka(x, y)
            if save_to_disk:
                np.save(
                    str(Path(cka_dir, f"cka_{split_name}_{'_'.join(seed_pair)}.npy")),
                    cka_values,
                )
            cka_matrices.append(cka_values)
            t.update()
    return cka_matrices


# todo: update
def cka_matrix(
    dirnames: List[str],
    # idx: np.ndarray,
    cka_dir: Union[str, Path],
    split_name: str,
    mode: str,
    save_to_disk: bool,
    activations_root: Path,
) -> List[np.ndarray]:
    if mode == "full":
        ckas = pairwise_cka_full(
            dirnames, cka_dir, split_name, activations_root, save_to_disk
        )
    elif mode == "diag":
        ckas = pairwise_cka_diag(
            dirnames, cka_dir, split_name, activations_root, save_to_disk,
        )
    else:
        raise ValueError(f"Unknown CKA mode: {mode}")
    return ckas
