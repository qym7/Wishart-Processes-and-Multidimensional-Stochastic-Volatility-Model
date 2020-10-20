import numpy as np

def cholesky_decomposition(A):
    '''

    :return:
    '''
    r = 0
    if A.shape[0]!=A.shape[1]:
        raise ValueError
    n = A.shape[0]
    for i in range(n):
        A_diag = np.array([A[k,k] for k in range(i, n)])
        q = np.where(A_diag==np.max(A_diag))[0][0] + i
        if A[q, q]>0:
            r += 1
            A[[i, q], :] = A[[q, i], :]
            A[:, [i, q]] = A[:, [q, i]]
            A[i, i] = np.sqrt(A[i, i])
            A[i+1:n, i] = A[i+1:n, i]/A[i,i] # why?
            for j in range(i,n):
                A[j:n, j] = A[j:n, j] - A[j:n, i]/A[j, i]

    return A
