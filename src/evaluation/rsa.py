import logging

import numpy as np
import rsatoolbox

log = logging.getLogger(__name__)


def get_rsa_cos(features_x, features_y):
    """
    get cos RSA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: cos RSA coefficient
    """
    # noinspection PyProtectedMember
    try:
        rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
        rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
        return rsatoolbox.rdm.compare(rdms1, rdms2, method='cosine').item()
    except np.core._exceptions._ArrayMemoryError:
        # try:
        rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x.astype(np.float16)))
        rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y.astype(np.float16)))
        result = rsatoolbox.rdm.compare(rdms1, rdms2, method='cosine').item()
        log.warning("Unable to allocate memory error in calculating RSA, retrying with float 16")
        return result
        # except np.core._exceptions._ArrayMemoryError:
        #     log.warning(
        #         "Unable to allocate memory error in calculating RSA, retry with float 16 failed, returning value 0")
        #     return 0


def get_rsa_corr(features_x, features_y):
    """
    get corr RSA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: corr RSA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    return rsatoolbox.rdm.compare(rdms1, rdms2, method='corr').item()


def get_rsa_corr_cov(features_x, features_y):
    """
    get corr_cov RSA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: corr_cov RSA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    return rsatoolbox.rdm.compare(rdms1, rdms2, method='corr_cov').item()


def get_rsa_tau_a(features_x, features_y):
    """
    get tau_a RSA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: tau_a RSA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    return rsatoolbox.rdm.compare(rdms1, rdms2, method='tau-a').item()


def get_rsa_rho_a(features_x, features_y):
    """
    get the rho_a RSA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return:rho_a RSA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    return rsatoolbox.rdm.compare(rdms1, rdms2, method='rho-a').item()
