import numpy as np
from numpy.random.mtrand import standard_normal
import gcm


def test_pass():
    np.seed(1)
    n = 250
    z = 4 * np.random.standard_normal(n)
    x = 2 * np.sin(z) + np.random.standard_normal(n)
    y = 2 * np.sin(z) + np.random.standard_normal(n)
    y2 = 2 * np.sin(z) + x + np.random.standard_normal(n)
    result = gcm.gcm(x, y, z, regr_method='gam')
    assert result


def test_fail():
    np.seed(1)
    n = 250
    z = 4 * np.random.standard_normal(n)
    x = 2 * np.sin(z) + np.random.standard_normal(n)
    y = 2 * np.sin(z) + np.random.standard_normal(n)
    y2 = 2 * np.sin(z) + x + np.random.standard_normal(n)
    result = gcm.gcm(x, y2, z, regr_method='gam')
    assert not result