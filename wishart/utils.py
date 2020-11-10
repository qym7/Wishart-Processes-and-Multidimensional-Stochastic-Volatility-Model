import numpy as np

def decompose_cholesky(M):
    '''
    :param M: a symmetric positive semi-definite matrix
    :return:
        * A: lower triangular part of Cholesky decomposition
        * c: lower triangular part of Cholevsky decomposition
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

def diag(M):
    '''
    :param M: a symmetric matrix
    :return:
        * D: the corresponding diagonal matrix of M
        * V: the matrix composed by all eigenvalues of M
    '''
    D, V = np.linalg.eig(M)
    D = np.diag(D)

    return D, V

def exp(M):
    '''
    :param M: a symmetric matrix
    :return: exp(M)
    '''
    d, v = diag(M)
    return v.dot(np.exp(d)).dot(v.T)


def brownian(N, M, T):
    '''
    Function used to generate Brownian motion path.
    * Params:
        N : The number of pieces.
        M : The number of paths to generate.
        T : Ending time.
    * Return:
        W : The brownian motion paths. Of shape (M, N+1).
    '''
    dT = T/N
    
    # Define Z, a matrix of shape M*N.
    Z = np.random.normal(size=(M, N))
    # Add a column of zeros to Z.
    Z = np.concatenate([np.zeros((M,1)), Z], axis=1)
    # Calculate W, the matrix of M discrete Brownian motions. Each row is a Brownian motion.
    W = np.cumsum(np.sqrt(dT)*Z, axis=1)
    
    return W