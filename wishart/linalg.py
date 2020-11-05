import numpy as np

def decompose_cholesky(M):
    '''
    :param M: a symmetric positive semi-definite matrix
    :return:
        * A: lower triangular part of cholesky decomposition
        * c: lower triangular part of cholevsky decomposition
        * k: M(n-r, r)
        * p: permutation matrix
        * r: range of matrix M
    '''
    r = 0
    A = np.array(M).copy()
    assert len(A.shape)==2 and A.shape[0]==A.shape[1]
    n = A.shape[0]
    p = np.eye(n)
    diag = np.diag(A)
    for i in range(n):
        A_diag = diag[i:]
        q = np.argmax(A_diag)+i
        if diag[q] > 0:
            r += 1
            p[[i, q], :] = p[[q, i], :]
            A[[i, q], :] = A[[q, i], :]
            A[:, [i, q]] = A[:, [q, i]]
            A[i, i] = np.sqrt(A[i, i])
            A[i+1:n, i] = A[i+1:n, i]/A[i,i]
            for j in range(i+1,n):
                A[j:n, j] = A[j:n, j] - A[j:n, i]*A[j, i]

    A = np.tril(A, k=1)
    c = A[:r, :r]
    k = A[r:, :r]

    return c, k, p, r
