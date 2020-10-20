import numpy as np

def cholesky_decomposition(M):
    '''
    :param M: a symmetric positive semi-definite matrix
    :return:
        * A: lower triangular part of cholevsky decomposition
        * p: permutation matrix
        * r: range of matrix M
    '''
    r = 0
    A = M.copy()
    if A.shape[0]!=A.shape[1]:
        raise ValueError
    n = A.shape[0]
    p = np.eye(n)
    for i in range(n):
        A_diag = np.array([A[k,k] for k in range(i, n)])
        q = np.where(A_diag==np.max(A_diag))[0][0] + i
        if A[q, q]>0:
            r += 1
            p[[i, q], :] = p[[q, i], :]
            A[[i, q], :] = A[[q, i], :]
            A[:, [i, q]] = A[:, [q, i]]
            A[i, i] = np.sqrt(A[i, i])
            A[i+1:n, i] = A[i+1:n, i]/A[i,i]
            for j in range(i+1,n):
                A[j:n, j] = A[j:n, j] - A[j:n, i]*A[j, i]

    for i in range(0, n):
        for j in range(i+1, n):
            A[i, j] = 0

    return A, p, r