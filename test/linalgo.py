import numpy as np

from wishart import linalg

def test_deposition():
    a = np.array([[1,2,3],[2,5,6],[3,6,9]])
    c,k,p,r = linalg.decompose_cholesky(a)
    print(c)
    print(k)
    print(p)
    print(r)

test_deposition()