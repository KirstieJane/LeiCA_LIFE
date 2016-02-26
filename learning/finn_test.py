def r_to_z(r):
    r = np.atleast_1d(r)
    r[r == 1] = 1 - 1e-15
    r[r == -1] = -1 + 1e-15
    return np.arctanh(r)


def get_sig_edges(X, y, p_thresh=.01):
    '''
    input vecotrized data (e.g. matrix) of shape n x p and
    target (e.g. age) of shape n,
    '''
    import numpy as np
    from scipy.stats import pearsonr

    r = np.zeros_like(X[0, :])
    p = np.zeros_like(X[0, :])

    for i in range(X.shape[1]):
        r[i], p[i] = pearsonr(y, X[:, i])

    sig_ind = {}
    strength = {}
    sig_ind['pos'] = (r >= 0) & (p <= p_thresh)
    sig_ind['neg'] = (r < 0) & (p <= p_thresh)

    z = r_to_z(r)
    strength['pos'] = z[sig_ind['pos']].sum()
    strength['neg'] = z[sig_ind['neg']].sum()
    return strength


import numpy as np

p_thresh = .001
X = np.random.randn(2500, 20910)
y = np.random.randn(2500, )
get_sig_edges(X, y, p_thresh)
