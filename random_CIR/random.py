'''
Defining Generalized Gaussian distribution, and non-central chi-square distribution.
19 Oct. 2020.
Benxin ZHONG
'''

def general_normal(q, size=1):
    '''
    Function to generate the samples of generalised Gaussian distribution.
    * Param(s):
        q : The parameter q in N(0, 1, q). Positive integer.
        size : Size of the generated sample. Shall be an integer, or an 1d array/tuple of integers.
    '''
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
    
    
def chi_2(p, q, lam=0.0, size=1):
    '''
    Function used to generate the samples of Chi-square distribution, with specified params nu and lambda,
    and the given size.
    * Params:
        p, q : Indicating the degree of freedom by nu = p/q. Positive integers.
        lam: Default is 0. The center. Non-negative real number.
        size : Default is 1. Size of the generated sample. Shall be an integer, or an 1d array/tuple of integers.
    '''
    size = np.array(size, dtype=int)
    if size.size == 1: #i.e., the required size is an integer.
        num = size
    else:
        assert len(size.shape) == 1 
        num =1
        for x in size:
            num = num*x
    assert (num>0)
    
    # Examin p and q.
    assert isinstance(p, int) and isinstance(q, int)
    assert p>0 and q>0
    d = np.gcd(p, q)
    p = int(p/d)
    q = int(q/d)
    
    # Generate the generalized Gaussian variables.
    lst_Z = general_normal(q=2*q, size=(num, p))
    # Calculate the number of standard Gaussian required.
    assert lam>=0
    if lam == 0:
        lst_N = np.zeros(num, dtype=int)
    else:
        lst_N = np.random.poisson(lam=lam/2, size=num)
        lst_N = lst_N.astype(int)
    # Generate Chi-square variables.
    lst_X = []
    for i in range(num):
        Z = lst_Z[i]
        N = lst_N[i]
        Y = np.random.normal(size=N)
        X = np.sum(np.power(Z, 2*q)) + np.sum(Y*Y)
        lst_X.append(X)
    # reshape
    lst_X = np.array(lst_X).reshape(size)
    return lst_X