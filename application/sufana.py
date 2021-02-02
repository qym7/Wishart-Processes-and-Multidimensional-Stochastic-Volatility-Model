"""
Implementation of the Gourieroux-Sufana model
"""

import numpy as np

import wishart
from wishart import utils


class GS_model:
    """
    The Gourieroux-Sufana model class.
    GS model defines the following process:
    dS_t = rS_t + (\sqrt(X_t) dB_t)^T S_t,
    dX_t = (\alpha a^Ta + bX_t + X_tb^T)dt + (\sqrt(X_t)dW_ta + a^TdW_t^T\sqrt(X_t))
    """
    def __init__(self, S0, r, X0, alpha, a, b):
        """
        * Params:
           S0 : d-dim vector, the initial assert value.
           r : real num, the interest rate.
           X0 : semi-pos-def d-dim matrix, the initial vol.
           alpha : real num > d-1.
           a : d-dim matrix.
           b : d-dim matrix.
        """
        self.S0 = S0
        self.r = r
        self.d = len(S0)
        assert X0.shape == (self.d, self.d)
        self.w_gen = wishart.Wishart(X0, alpha, b=b, a=a)
        
    def __call__(self, num, N, T, ret_vol=False, method='2', num_int=2000, **kwargs):
        if method == '2':
            return self.gen(num=num, N=N, T=T, ret_vol=ret_vol, num_int=num_int, **kwargs)
        elif method == 'euler' or method == '4':
            return self.euler(num, N, T, ret_vol, **kwargs)
    
    def gen(self, num, N, T, ret_vol, num_int, **kwargs):
        S = np.zeros((num, N, self.d))
        X = np.zeros((num, N, self.d, self.d))
        h = T/N
        qh = utils.integrate(h, self.w_gen.b, self.w_gen.a, self.d, num_int=num_int)

        x = np.expand_dims(self.w_gen.x, axis=0)
        s = self.S0
        for i in range(N):
            s = self.step_S(num=num, h=h/2, x=x, S0=s)
            x = self.step_W(num=num, h=h, x=x, qh=qh, method='exact', **kwargs)
            s = self.step_S(num=num, h=h/2, x=x, S0=s)
            S[:, i] = s
            X[:, i] = x
            
        if ret_vol:
            return S, X
        else:
            return S

    def step_W(self, num, h, x, **kwargs):
        if x.shape[0] == 1:
            num = num
            x = x.repeat(num, axis=0)
        else:
            num = x.shape[0]

        X = np.zeros((num, x.shape[1], x.shape[2]))
        for i in range(num):
            X[i] = self.w_gen(num=1, N=1, T=h, x=x[i], **kwargs)

        return X

    def step_S(self, num, h, x, S0):
        '''
        return S of shape: num * d
        '''
        if x.shape[0] == 1:
            num = num
        else:
            num = x.shape[0]

        G = np.random.normal(size=(num, self.d, 1))
        diag_x = np.diagonal(x, axis1=1, axis2=2)  # Take the diagonals of x.
        tmp_1 = (self.r - diag_x / 2) * h  # Of shape (num, d).
        std = np.sqrt(h) * wishart.cholesky(x)
        if x.shape[0] == 1:
            std = std.repeat(num, axis=0)
        tmp_2 = np.matmul(std, G)  # Of shape (num, d, 1).
        tmp_2 = tmp_2.reshape(num, self.d)
        S = S0 * np.exp(tmp_1 + tmp_2)  # Of shape (num, d).

        return S

    def euler(self, num, N, T, ret_vol, trace=True, **kwargs):
        X = self.w_gen(num=num, N=N, T=T, method='2', trace=trace, **kwargs)
        num = X.shape[0]
        h = T/N
        sqrt_h = np.sqrt(h)
        S = np.zeros((num, N+1, self.d))
        S[:, 0] = self.S0
        G = np.random.normal(size=(num, N, self.d, 1))

        for i in range(1, N+1):
            s0 = S[:, i-1].reshape(num, self.d, 1)
            x = X[:, i-1]  # Of shape (num, d, d).
            std = sqrt_h * wishart.cholesky(x)
            tmp = np.matmul(std, G[:, i-1])  # Of shape (num, d, 1).
            tmp = 1 + self.r * h + tmp
            s1 = s0 * tmp  # S_t(1+r dt + \sqrt(X_t)dB_t).
            S[:, i] = s1.reshape(num, self.d)

        if ret_vol:
            return S, X
        else:
            return S
