import numpy as np
import scipy.linalg
import sys 
sys.path.append('..')
import sampling


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
    A = np.array(M.copy(), dtype=np.float64)
    assert len(A.shape)==2 and A.shape[0]==A.shape[1]
    n = A.shape[0]
    p = np.eye(n)
    diag = np.diag(A)
    for i in range(n):
        A_diag = diag[i:]
        q = np.argmax(A_diag) + i
        if diag[q] > 0:
            r += 1
            p[[i, q], :] = p[[q, i], :]
            A[[i, q], :] = A[[q, i], :]
            A[:, [i, q]] = A[:, [q, i]]
            A[i, i] = np.sqrt(A[i, i])
            A[i+1:n, i] = A[i+1:n, i]/A[i,i]
            for j in range(i+1,n):
                A[j:n, j] = A[j:n, j] - A[j:n, i]*A[j, i]

    A = np.tril(A, k=0)
    c = A[:r, :r]
    k = A[r:, :r]

    p = np.array(p, dtype=int)
    return c, k, p, r


def cholesky(M):
    M = np.array(M, dtype=float)
    shape = M.shape
    assert shape[-1] == shape[-2]
    d = shape[-1]
    M_ = M.reshape(-1, d, d)
    lst_c = []
    for m in M_:
        c, k, p ,r = decompose_cholesky(m)
        p = np.array(p, dtype=float)
        b = np.zeros(m.shape, dtype=float)
        b[:r, :r] = c
        b[r:, :r] = k
        lst_c.append(np.matmul(p.T, b))
    if len(shape)==2:
        return lst_c[0]
    else:
        lst_c = np.array(lst_c).reshape(shape)
        return lst_c


def is_sdp(M, debug=False):
    '''
    :param M: a matrix
    :return:
        * boolean: if M is symetric, semi definite positive
    '''
    is_pos = True
    c, k, p, r = decompose_cholesky(M)
    b = np.zeros(M.shape)
    b[:r, :r] = c
    b[r:, :r] = k
    err = p.T.dot(b.dot(b.T)).dot(p) - M
    if debug:
        return err
    return np.isclose(err, 0).all()


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


def brownian(N, M, T, method="exact"):
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
    if method == "exact":
        Z = np.random.normal(size=(M, N))
    elif method == "2" or method == 2:
        Z = sampling.bounded_gauss(size=(M, N))
    elif method == "3" or method == 3:
        Z = sampling.bounded_gauss(size=(M, N), order=3)
    else:
        raise ("Method is not supported, please choose from [exact, 2, 3]")
    # Add a column of zeros to Z.
    Z = np.concatenate([np.zeros((M,1)), Z], axis=1)
    # Calculate W, the matrix of M discrete Brownian motions. Each row is a Brownian motion.
    W = np.cumsum(np.sqrt(dT)*Z, axis=1)
    
    return W


def integrate(T, b, a, d, num_int=200):
    assert b.shape == (d, d) and a.shape == (d, d)
    dt = T / num_int
    lst_t = np.arange(num_int) * dt
    dqt = np.array([scipy.linalg.expm(t * b).dot(a.T) for t in lst_t])
    dqt = np.array([dqt[i].dot(dqt[i].T) for i in range(num_int)])

    return np.sum(dqt, axis=0) * dt

def brownian_squrare(T, dimension, n_steps=1):
    W = np.zeros((n_steps + 1, dimension, dimension))
    W[1:] = np.random.randn((n_steps, dimension, dimension))*np.sqrt(T/n_steps)

    return W.cumsum(axis=0)
