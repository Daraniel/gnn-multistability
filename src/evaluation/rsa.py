import rsatoolbox


def get_rsa_cos(features_x, features_y):
    """
    get the mean absolute Procrustes coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    rsatoolbox.rdm.compare(rdms1, rdms2, method='cosine')


def get_rsa_corr(features_x, features_y):
    """
    get the mean absolute Procrustes coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    rsatoolbox.rdm.compare(rdms1, rdms2, method='corr')


def get_rsa_corr_cov(features_x, features_y):
    """
    get the mean absolute Procrustes coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    rsatoolbox.rdm.compare(rdms1, rdms2, method='corr_cov')


def get_rsa_tau_a(features_x, features_y):
    """
    get the mean absolute Procrustes coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    rsatoolbox.rdm.compare(rdms1, rdms2, method='tau-a')


def get_rsa_rho_a(features_x, features_y):
    """
    get the mean absolute Procrustes coefficient for different features
    :param features_x: x features
    :param features_y: y features
    :return: mean absolute value of CCA coefficient
    """
    rdms1 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_x))
    rdms2 = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(features_y))
    rsatoolbox.rdm.compare(rdms1, rdms2, method='rho-a')
