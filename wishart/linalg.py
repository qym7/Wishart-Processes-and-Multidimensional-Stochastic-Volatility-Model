import numpy as np

def decompose_cholesky(M):
    '''
    :param M: a symmetric positive semi-definite matrix
    :return:
<<<<<<< HEAD
        * A: lower triangular part of cholesky decomposition
=======
        * c: lower triangular part of cholevsky decomposition
        * k: M(n-r, r)
>>>>>>> f966e8e83473374876a4a07c4e139a49415faeeb
        * p: permutation matrix
        * r: range of matrix M
    '''
    r = 0
#     A = M.copy()
    A = np.array(M).copy()
    assert len(A.shape)==2 and A.shape[0]==A.shape[1]
#     if A.shape[0]!=A.shape[1]:
#         raise ValueError
    n = A.shape[0]
    p = np.eye(n)
    diag = np.diag(A)
    for i in range(n):
#         A_diag = np.array([A[k,k] for k in range(i, n)])
        A_diag = diag[i:]
#         q = np.where(A_diag==np.max(A_diag))[0][0] + i
        q = np.argmax(A_diag)+i
#         if A[q, q]>0:
        if diag[q] > 0:
            r += 1
            p[[i, q], :] = p[[q, i], :]
            A[[i, q], :] = A[[q, i], :]
            A[:, [i, q]] = A[:, [q, i]]
            A[i, i] = np.sqrt(A[i, i])
            A[i+1:n, i] = A[i+1:n, i]/A[i,i]
            for j in range(i+1,n):
                A[j:n, j] = A[j:n, j] - A[j:n, i]*A[j, i]

<<<<<<< HEAD
#     for i in range(0, n):
#         for j in range(i+1, n):
#             A[i, j] = 0
    A = np.tril(A, k=1)
=======
    c = np.zeros((r, r))
    k = np.zeros((n-r, r))
    for i in range(0, r):
        for j in range(0,i+1):
            c[i, j] = A[i,j]
    for i in range(0, r):
        for j in range(r, n):
            k[j-r, i] = A[j, i]
>>>>>>> f966e8e83473374876a4a07c4e139a49415faeeb

    return c, k, p, r
