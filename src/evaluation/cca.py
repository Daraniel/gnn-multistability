import logging

import numpy as np
from sklearn.cross_decomposition import CCA


log = logging.getLogger(__name__)

def get_cca(features_x, features_y):
    """
    get the mean absolute CCA coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    try:
        cca = CCA(n_components=features_x.shape[1], max_iter=200)
        cca.fit(features_x, features_y)

        return np.mean(np.abs(cca.coef_)).item()
    except:
        log.warning("Error in calculating CCA, returning value 0")
        return 0
