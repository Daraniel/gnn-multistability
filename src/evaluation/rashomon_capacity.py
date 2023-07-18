""" code based on https://github.com/HsiangHsu/rashomon-capacity/blob/main/utils/capacity.py"""
import logging
import multiprocessing
from functools import partial
from itertools import islice
from multiprocessing import Pool
from threading import Condition
from types import SimpleNamespace

import numpy as np

log = logging.getLogger(__name__)

def blahut_arimoto(Pygw, log_base=2, epsilon=1e-12, max_iter=1e3):
    """
    Performs the Blahut-Arimoto algorithm to compute the channel capacity
    given a channel P_ygx.

    Parameters
    ----------
    Pygw: shape (m, c).
        transition matrix of the channel with m inputs and c outputs.
    log_base: int.
        base to compute the mutual information.
        log_base = 2: bits, log_base = e: nats, log_base = 10: dits.
    epsilon: float.
        error tolerance for the algorithm to stop the iterations.
    max_iter: int.
        number of maximal iteration.
    Returns
    -------
    Capacity: float.
        channel capacity, or the maximum information it can be transmitted
        given the input-output function.
    pw: array-like.
        array containing the discrete probability distribution for the input
        that maximizes the channel capacity.
    loop: int
        the number of iteration.
    resource: https://sites.ecse.rpi.edu/~pearlman/lec_notes/arimoto_2.pdf
    """
    ## check inputs
    # assert np.abs(Pygw.sum(axis=1).mean() - 1) < 1e-6
    # assert Pygw.shape[0] > 1

    m = Pygw.shape[0]
    c = Pygw.shape[1]
    Pw = np.ones((m)) / m
    for cnt in range(int(max_iter)):
        ## q = P_wgy
        q = (Pw * Pygw.T).T
        q = q / q.sum(axis=0)

        ## r = Pw
        r = np.prod(np.power(q, Pygw), axis=1)
        r = r / r.sum()

        ## stoppung criteria
        if np.sum((r - Pw) ** 2) / m < epsilon:
            break
        else:
            Pw = r

    ## compute capacity
    capacity = 0
    for i in range(m):
        for j in range(c):
            ## remove negative entries
            if r[i] > 0 and q[i, j] > 0:
                capacity += r[i] * Pygw[i, j] * np.log(q[i, j] / r[i])

    capacity = capacity / np.log(log_base)
    return capacity, r, cnt + 1


def compute_capacity(likelihood, log_base=2, epsilon=1e-12, max_iter=1e3):
    # m, n, c = likelihood.shape[0], likelihood.shape[1], likelihood.shape[2]
    n, m, c = likelihood.shape[0], likelihood.shape[1], likelihood.shape[2]
    cores = min(16, multiprocessing.cpu_count() - 1)
    it = iter(range(n))
    ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
    # compute in parallel
    cvals = SimpleNamespace(_value=[None])
    timeout_seconds = 1200  # HINT: start with small wait time to prevent getting stuck due to error
    max_tries = 50
    for i in range(max_tries):
        with Pool(cores) as p:
            condition = Condition()
            condition.acquire()
            # cvals = (p.map(blahut_arimoto, [likelihood[:, ix, :].reshape((m, c)) for ix in ln]))
            cvals = (p.map_async(partial(blahut_arimoto, log_base=log_base, epsilon=epsilon, max_iter=max_iter),
                                 [likelihood[ix] for ix in ln]))
            cvals.wait(timeout_seconds)
            p.terminate()
            if None not in cvals._value:
                break
            log.warning("timeout in calculating rashomon capacity, retrying")
            timeout_seconds *= 3  # HINT: increase wait time
    if None in cvals._value:
        log.error("Failed to calculate rashomon capacity")
        return []
    capacity = np.array([v[0] for v in cvals._value])
    return capacity

# def compute_capacity2(likelihood):
#     m, n, c = likelihood.shape[0], likelihood.shape[1], likelihood.shape[2]
#     # n, m, c = likelihood.shape[0], likelihood.shape[1], likelihood.shape[2]
#     cores = multiprocessing.cpu_count() - 1
#     it = iter(range(n))
#     ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
#     # compute in parallel
#     with Pool(cores) as p:
#         cvals = (p.map(blahut_arimoto, [likelihood[:, ix, :].reshape((m, c)) for ix in ln]))
#         # cvals = (p.map(blahut_arimoto, [likelihood[ix] for ix in ln]))
#     capacity = np.array([v[0] for v in cvals])
#     return capacity
