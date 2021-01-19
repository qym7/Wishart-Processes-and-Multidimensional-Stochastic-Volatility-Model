import numpy as np
import scipy.linalg

from wishart import utils
from cir import CIR
import sampling

class Wishart():
    def __init__(self, x, alpha, a=None, b=None):
        '''
        :param x: np.array of shape (d, d). The initial state of the process.
        :param alpha:
        :param b:
        :param a:
        '''
        self.x = x
        assert utils.is_sdp(x)
        self.c, self.k, self.p, self.r = utils.decompose_cholesky(self.x[1:, 1:])

        self.d = x.shape[0]
        assert alpha >= self.d - 1
        self.alpha = alpha
        if b is None:
            self.b = np.zeros((self.d, self.d))
        else:
            assert b.shape[0] == self.d and b.shape[1] == self.d
            self.b = b
        if a is None:
            self.a = np.eye(self.d)
        else:
            assert a.shape[0] == self.d and a.shape[1] == self.d
            self.a = a

    def __call__(self, T, x=None, N=1, num=1, method="exact", **kwargs):
        if x is not None:
            assert utils.is_sdp(x)

        if method == "euler":
            return self.euler(T=T, x=x, N=N, num=num, **kwargs)
        else:
            return self.wishart(T, N=N, x=x, num=num, method=method, **kwargs)

    def hr(self, u, r=None, c=None, k=None):
        '''
        :param u: an np.array of shape (N+1, r+1)
        :return: an np.array of shape (N+1, self.d, self.d)
        '''
        if r is None:
            r = self.r
        if c is None:
            c = self.c
        if k is None:
            k = self.k
        m1 = np.eye(self.d)
        m1[1:r+1, 1:r+1] = c
        m1[r+1:self.d, 1:r+1] = k
        m3 = m1.copy().T
        
        ut = u
        m2 = np.zeros((self.d, self.d))
        m2[0, 0] = ut[0] + np.sum(ut[1:r+1] * ut[1:r+1])
        m2[0, 1:r+1] = ut[1:r+1]
        m2[1:r+1, 0] = ut[1:r+1]
        m2[1:r+1, 1:r+1] = np.eye(r)
        
        return np.matmul(m1, np.matmul(m2, m3))

    def wishart_e(self, T, N, num=1, x=None, method="exact", **kwargs):
        '''
        :param T: Non-negative real number.
        :param N: Positive integer. The number of discrete time points.
        :param x: (d, d) or (num, d, d). The initial value(s). If x is None, self.x is used.
        :return: an np.array of shape (num, N+1, self.d, self.d).
        '''
        assert T >= 0 and N > 0
        if x is None:
            x = self.x
        assert x.shape == (num, self.d, self.d) or x.shape == (self.d, self.d)
        h = T/N
        
        X_proc = np.zeros((num, N+1, self.d, self.d))
        X_proc[:, 0] = x
        for i in range(1, N+1):
            for n in range(num):
                X_proc[n, i] = self.step_wishart_e(h, x=X_proc[n, i-1], method=method)
        
        if 'trace' in kwargs and kwargs['trace']:
            return X_proc
        else:
            return X_proc[:, -1]
        
    
    def step_wishart_e(self, h, x, method='exact'):
#         x = np.array(x)
#         assert len(x.shape)==2 and x.shape[0]==self.d and x.shape[1]==self.d
#         np.linalg.cholesky(x)
        c, k, p, r = utils.decompose_cholesky(x[1:, 1:])
        pi = np.eye(self.d)
        pi[1:, 1:] = p
        xp = np.matmul(pi, np.matmul(x, pi.T))
        u = np.zeros(r+1)
        u[1:] = np.linalg.inv(c).dot(xp[0, 1:r+1])
        u[0] = xp[0, 0] - (u[1:]*u[1:]).sum()
        if not u[0]>=0:
            if np.isclose(u[0], 0):
                u[0] = 0
            else:
                print(f'Err! u[0]={u[0]}.')
        
        cir_u0 = CIR(k=0, a=self.alpha-r, sigma=2, x0=u[0])  # The CIR generator.
        
        u0 = cir_u0(T=h, n=1, num=1, method=method)[0, -1]
        U = np.zeros(r+1)
        U[0] = u0
#         U[1:] = u[1:] + np.sqrt(h) * np.random.normal(size=r)
        U[1:] = u[1:] + np.sqrt(h) * sampling.random.gauss(size=r, method=method)
        Xt = self.hr(u=U, r=r, c=c, k=k)
        Xt = np.matmul(pi.T, np.matmul(Xt, pi))
        return Xt


    def wishart_i(self, T, n, N=1, num=1, x=None, method="exact", **kwargs):
        '''
        :param T: Non-negative real number.
        :param n: Positive integer. The n of I_d^n.
        :param N: Positive integer. The number of discrete time points.
        :param x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :return: an np.array of shape (num, self.d, self.d).
        '''
        assert T >= 0 and N > 0
        if x is None:
            x = self.x
        assert x.shape == (num, self.d, self.d) or x.shape == (self.d, self.d)
        h = T/N
        
        X_proc = np.zeros((num, N+1, self.d, self.d))
        X_proc[:, 0] = x
        for i in range(1, N+1):
            X_proc[:, i] = self.step_wishart_i(h, n, x=X_proc[:, i-1], num=num, method=method)
        
        if 'trace' in kwargs and kwargs['trace']:
            return X_proc
        else:
            return X_proc[:, -1]
            
    
    def step_wishart_i(self, h, n, x, num=1, method='exact'):
        x = np.array(x)
        assert x.shape == (self.d, self.d) or x.shape==(num, self.d, self.d)
#         assert len(x.shape)==2 and x.shape[0]==self.d and x.shape[1]==self.d
        y = np.zeros((num, self.d, self.d))
        y[:] = x
        
        for k in range(n):
            if k == 0:
                Y = np.array([self.step_wishart_e(h=h, x=y[i], method=method) for i in range(num)])
                y = Y
            else:
                p = np.eye(self.d)
                p[0, 0] = p[k, k] = 0
                p[k, 0] = p[0, k] = 1
                y = np.matmul(p, np.matmul(y, p))
                Y = np.array([self.step_wishart_e(h=h, x=y[i], method=method) for i in range(num)])
                y = np.matmul(p, np.matmul(Y, p))
        
        if num==1:
            y = y.reshape(x.shape)
            
        return y
            
        

    def wishart(self, T, x=None, N=1, num=1, method="exact", **kwargs):
        '''
        :param T: Non-negative real number.
        :param N: Positive integer. The number of discrete time points.
        :param x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :param num: num of simulations
        
        method: exact, 2, 3 or euler.
        :return: an np.array of shape (num, self.d, self.d).
        '''
        if x is None:
            x = self.x
        a = self.a
        b = self.b
        if 'num_int' in kwargs:
            num_int = kwargs['num_int']
        else:
            num_int = 200
        
        h = T/N
        # Here we shall find a method to calculate q.
        qh = utils.integrate(h, b, a, self.d, num_int=num_int)
        # Calculate cholesky decomposition of qh/h and a^Ta.
        theta = utils.cholesky(qh/h)
        c, k, p, n = utils.decompose_cholesky(qh/h)
        # Build theta_t.
        theta = np.eye(self.d)
        theta[:n, :n] = c
        theta[n:, :n] = k
#         theta = np.linalg.inv(p).dot(theta)
        theta = np.matmul(np.linalg.inv(p), theta)
        m = scipy.linalg.expm(b*h)
        theta_inv = np.linalg.inv(theta)
        tmp_fac = np.matmul(theta_inv, m)
        
        
        X_proc = np.zeros((num, N+1, self.d, self.d))
        X_proc[:, 0] = x
        for i in range(1, N+1):
            x = X_proc[:, i-1]
            x_tmp = np.matmul(tmp_fac, np.matmul(x, tmp_fac.T))
#             Y = self.wishart_i(T=h, n=n, N=1, num=num, x=x_tmp, method=method)
            Y = self.step_wishart_i(h=h, n=n, x=x_tmp, num=num, method=method)
            X = np.matmul(theta, np.matmul(Y, theta.T))
            X_proc[:, i] = X
        
        if 'trace' in kwargs and kwargs['trace']:
            return X_proc
        else:
            return X_proc[:, -1]
    
    def euler(self, T, x=None, N=100, num=1, **kwargs):
        '''
        Euler discretization scheme.
        return:
            X, of shape (num, d, d). 
                if trace=True is indicated, X is of shape (num, N+1, d, d).
        '''
        if x is None:
            x = self.x
        else:
            assert x.shape == (self.d, self.d)
        a = self.a
        b = self.b
        alpha_bar = self.alpha * a.T.dot(a)
        
        dt = T/N
        dW = np.sqrt(dt) * np.random.normal(size=(num, N, self.d, self.d))
        X = np.zeros((num, N+1, self.d, self.d))
        X[:, 0] = x
        for i in range(1, N+1):
            X0 = X[:, i-1]
            # Calculate \bar{alpha} + B(X0).
            xbt = np.matmul(X0, b.T)
            tmp_t = alpha_bar + (xbt + xbt.transpose((0,2,1)))
            # Calculate sqrt(X0).
            sqrt_x0 = np.array([scipy.linalg.sqrtm(X0[j]) for j in range(num)])
            sqrt_x0 = sqrt_x0.real # Incase where X0 is not non-neg definit.
            # Calculate sqrt{X0}dWa
            dWt = dW[:, i-1]
            tmp_W = np.matmul(np.matmul(sqrt_x0, dWt), a)
            tmp_W = tmp_W + tmp_W.transpose((0,2,1))
            X1 = X0 + tmp_t + tmp_W
            X[:, i] = X1
        
        if 'trace' in kwargs and kwargs['trace']:
            return X
        else:
            return X[:, -1]
        
    def character(self, T, v, x=None, num_int=200):
        '''
        Function used to calculate E[exp(Tr(vX_T))].
        * Params:
            v : A sequence of matrices, of shape (num, d, d).
        * Return:
            char : A sequence of chars, of shape (num, )
        '''
        if x is None:
            x = self.x
        qt = utils.integrate(T, self.b, self.a, d=self.d, num_int=num_int)
        mt = scipy.linalg.expm(T*self.b)
        qtv = np.matmul(qt, v) # Of shape (num, d, d).
        Idqtv = (np.eye(self.d) - 2*qtv)
        tmp = np.linalg.inv(Idqtv)
        tmp = np.matmul(v, tmp)
        tmp = np.matmul(tmp, mt.dot(x).dot(mt.T))
        nom = np.exp(np.trace(tmp, axis1=1, axis2=2)) # of shape (num, )
        
        det = np.linalg.det(Idqtv)
        den = np.power(det, self.alpha/2)
        
        return nom/den
