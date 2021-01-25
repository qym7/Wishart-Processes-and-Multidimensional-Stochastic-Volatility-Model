"""
Implementation of the Gourieroux-Sufana model
"""

import numpy as np
import scipy.stats as stats
import scipy.linalg

import cir
import wishart

class GS_model:
    """
    The Gourieroux-Sufana model class.
    GS model defines the following process:
    dS_t = rS_t + (\sqrt(X_t) dB_t)^T S_t,
    dX_t = (\alpha a^Ta + bX_t + X_tb^T)dt + (\sqrt(X_t)dW_ta + a^TdW_t^T\sqrt(X_t))
    """
    def __init__(self, S0, r, X0, alpha, a, b):
        '''
        * Params:
           S0 : d-dim vector, the initial assert value.
           r : real num, the interest rate.
           X0 : semi-pos-def d-dim matrix, the initial vol.
           alpha : real num > d-1.
           a : d-dim matrix.
           b : d-dim matrix.
        '''
        self.S0 = S0
        self.r = r
        self.d = len(S0)
        assert X0.shape == (self.d, self.d)
        self.w_gen = wishart.Wishart(X0, alpha, b=b, a=a)
        
    def __call__(self, num, N, T, ret_vol=False, method='exact', **kwargs):
        return self.gen(num, N, T, ret_vol, method=method, **kwargs)
    
    def gen(self, num, N, T, ret_vol, method, **kwargs):
        h = T/N
        X = self.w_gen(num=num, N=N, T=T, trace=True, method=method, **kwargs)
        S = self.gen_S(h, X, method='exact')
            
        if ret_vol:
            return S, X
        else:
            return S
        
    def gen_S(self, h, X, method='exact'):
        num = X.shape[0]
        N = X.shape[1]-1
        sqrt_h = np.sqrt(h)
        S = np.zeros((num, N+1, self.d))
        S[:, 0] = self.S0
        G = np.random.normal(size=(num, N, self.d, 1))
        
        if method == 'exact':  # St_l = S0_l*exp[(r-x_ll/2)t+(\sqrt(X)B_t)_l].
            for i in range(1, N+1):
                s0 = S[:, i-1].reshape(num, self.d)  # Of shape (num, d).
                x = X[:, i-1]  # Of shape(num, d, d).
                diag_x = np.diagonal(x, axis1=1, axis2=2)  # Take the diagonals of x.
                tmp_1 = (self.r - diag_x/2) * h  # Of shape (num, d).
                std = sqrt_h * wishart.cholesky(x)
                tmp_2 = np.matmul(std, G[:, i-1])  # Of shape (num, d, 1).
                tmp_2 = tmp_2.reshape(num, self.d) 
                s1 = s0 * np.exp(tmp_1 + tmp_2)  # Of shape (num, d).
                S[:, i] = s1

        elif method == 2 or method == '2':
            for i in range(1, N+1):
                s0 = S[:, i-1].reshape(num, self.d, 1)
                x = X[:, i-1]  # Of shape (num, d, d).
                std = sqrt_h * wishart.cholesky(x)
                # \sqrt(X_t)dB_t
                tmp = np.matmul(std, G[:, i-1])  # Of shape (num, d, 1).
                tmp = 1 + self.r * h + tmp
                s1 = s0 * tmp  # S_t(1+r dt + \sqrt(X_t)dB_t).
                S[:, i] = s1.reshape(num, self.d)

        return S
