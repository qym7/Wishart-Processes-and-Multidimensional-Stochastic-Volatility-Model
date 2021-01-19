'''
Defining Generalized Gaussian distribution, and non-central chi-square distribution.
19 Oct. 2020.
Benxin ZHONG
'''
import numpy as np

def general_normal(q, size=None):
    '''
    Function to generate the samples of generalised Gaussian distribution.
    * Param(s):
        q : The parameter q in N(0, 1, q). Positive integer.
        size : Size of the generated sample. Shall be an integer, or an 1d array/tuple of integers.
    '''
    if size is None:
        size = 1
    # Check size.
    size = np.array(size, dtype=int)
    # Calculate the required number of sample instances.
    if size.size == 1: #i.e., the required size is an integer.
        num = size
    else:
        assert len(size.shape) == 1 
        num =1
        for x in size:
            num = num*x
    assert (num>0)
    # Calculate the number of U = [U1, ..., U_q] required.
    num_Z = int(np.ceil(num/q))
    
    # Generate.
    lst_Z = []
    while len(lst_Z) < num_Z:
        U = np.random.rand(q)
        U = 2*U - 1 # U ~ Uniform([-1, 1]).
        q_norm = np.linalg.norm(U, ord=q)
        if q_norm < 1: # This U satisfies the condistion.
            tmp = -2 * q * np.log(q_norm)
            numerator = np.power(tmp, 1/q) # The numerator.
            Z = U * (numerator/q_norm) # Z_i ~ N(0, 1, q), i.i.d..
            lst_Z.append(Z)
    # Build the required shape of samples. 
    lst_Zi = np.array(lst_Z).reshape(-1)
    assert len(lst_Zi) >= num
    X = (lst_Zi[:num]).reshape(size)
    return X

def chi_2(p, q, lam=0.0, size=None):
    '''
    Function used to generate the samples of Chi-square distribution, with specified params nu and lambda,
    and the given size.
    * Params:
        p, q : Indicating the degree of freedom by nu = p/q. Positive integers.
        lam: Default is 0. The center. Non-negative real number, 
             or an array of non-negative real numbers.
        size : Default is None. Size of the generated sample. Shall be an integer, 
               or an 1d array/tuple of integers. 
               If size is None, it is determined as the same shape of lam.
               If lam is an array, size MUST be the shape of lam.
    '''
    # Check lam.
    lam = np.array(lam)
    if lam.size > 1: # lam is an array.
        if size is None:
            size = np.array(lam.shape)
        else:
            size = np.array(size)
            assert (size==lam.shape).all()
        assert (lam>=0).all()
        lam = lam.reshape(-1)
    else: # lam is an integer.
        assert lam>=0
        if size is None:
            size = 1
    # Calculate the number of sample to generate.   
    size = np.array(size, dtype=int)
    if size.size == 1: #i.e., the required size is an integer.
        num = size
    else:
        assert len(size.shape) == 1 
        num = 1
        for x in size:
            num = num*x
    num = int(num)
    assert (num>0)
    if lam.size>1:
        assert lam.size==num
    
    # Examin p and q.
    assert isinstance(p, int) and isinstance(q, int)
    assert p>0 and q>0
    d = np.gcd(p, q)
    p = int(p/d)
    q = int(q/d)
    
    # Generate the generalized Gaussian variables.
    lst_Z = general_normal(q=2*q, size=(num, p))
    # Calculate the number of standard Gaussian required.
    lst_N = np.random.poisson(lam=lam/2, size=num)
    lst_N = lst_N.astype(int)
    # Generate Chi-square variables.
    lst_X = []
    for i in range(num):
        Z = lst_Z[i]
        N = lst_N[i]
        Y = np.random.normal(size=2*N)
        X = np.sum(np.power(Z, 2*q)) + np.sum(Y*Y)
        lst_X.append(X)
    # reshape
    lst_X = np.array(lst_X).reshape(size)
    return lst_X


def bounded_gauss(size=1, order=2):
    '''
    Function used to generate the required size of indep instances of Y,
    where P[Y=sqrt(3)] = P[Y=-sqrt(3)] = 1/6, and P[Y=0] = 2/3.
    '''
    U = np.random.uniform(size=size)
    Y = np.zeros_like(U)
    
    if order == 2:
        ind_pos = U < 1/6
        ind_neg = U > 5/6
        Y[ind_pos] = np.sqrt(3)
        Y[ind_neg] = -1 * np.sqrt(3)
    elif order == 3:
        # Seperate sqrt(3 + sqrt(6)) and sqrt(3 - sqrt(6))
        ind_pos = U < (np.sqrt(6)-2) / (2*np.sqrt(6))
        ind_neg = ~ind_pos
        Y[ind_pos] = np.sqrt(3 + np.sqrt(6))
        Y[ind_neg] = np.sqrt(3 - np.sqrt(6))
        # Genrate sign.
        sign = np.random.randint(2, size=size)*2 - 1
        Y = Y*sign
    else:
        raise ('Err, order shall be 2 or 3.')
    
    return Y

def gauss(size=1, method='exact'):
    if method == 'exact':
        return np.random.normal(size=size)
    elif method == '2' or method == 2:
        return bounded_gauss(size=size, order=2)
    elif method == '3' or method == 3:
        return bounded_gauss(size=size, order=3)
        