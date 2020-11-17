import numpy as np

from wishart import utils

def test_deposition():
    a = np.array([[1,2,3],[2,5,6],[3,6,9]])
    c,k,p,r = utils.decompose_cholesky(a)
    print(c)
    print(k)
    print(p)
    print(r)

test_deposition()