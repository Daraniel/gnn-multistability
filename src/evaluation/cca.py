from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cross_decomposition import CCA


def get_cca(features_x, features_y):
    """
    get the mean absolute CCA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    cca = CCA(n_components=features_x.shape[1])
    cca.fit(features_x, features_y)

    return np.mean(np.abs(cca.coef_)).item()
