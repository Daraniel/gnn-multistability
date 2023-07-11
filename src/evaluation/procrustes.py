from scipy.spatial import procrustes


def get_procrustes(features_x, features_y):
    """
    get Procrustes disparity for different features
    :param features_x: x features
    :param features_y: y features
    :return: Procrustes disparity
    """
    result = procrustes(features_x, features_y)
    return result[2]
