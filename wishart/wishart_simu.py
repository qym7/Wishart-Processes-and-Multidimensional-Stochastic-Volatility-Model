import numpy as np
import scipy.linalg
# from scipy.integrate import quad

from wishart import utils
from cir import CIR

class Wishart():
    def __init__(self, x, alpha, a=None, b=None):
        '''
        :param x: np.array of shape (d, d). The initial state of the process.
        :param alpha:
        :param b:
        :param a:
        '''
        try:
            np.linalg.cholesky(x)
        except:
            print("Err, x is not a symetric positive definite matrix.")
        self.x = x
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
        self.c, self.k, self.p, self.r = utils.decompose_cholesky(self.x[1:, 1:])

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
        
        lst_hru = []
        for i in range(u.shape[0]):
            ut = u[i]
            m2 = np.zeros((self.d, self.d))
            m2[0, 0] = ut[0]+np.sum(ut[1:r + 1] * ut[1:r + 1])
            m2[0, 1:r + 1] = ut[1:r + 1]
            m2[1:r + 1, 0] = ut[1:r + 1]
            m2[1:r+1, 1:r+1] = np.eye(r)

            lst_hru.append(m1.dot(m2).dot(m3))

        return np.array(lst_hru)

    def wishart_e(self, T, N, num=1, x=None, W=None):
        '''
        :param T: Non-negative real number.
        :param N: Positive integer. The number of discrete time points.
        :param x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :Param W: np.array of shape (r, N+1).
        :return: an np.array of shape (num, N+1, self.d, self.d).
        '''
        assert T >= 0 and N > 0
        # Check x.
        if x is None:
            x = self.x
            p = self.p
            c = self.c
            k = self.k
            r = self.r
        else:
            x = np.array(x)
            assert len(x.shape)==2 and x.shape[0]==self.d and x.shape[1]==self.d
            try:
                np.linalg.cholesky(x)
            except:
                print("Err, x is not a symetric positive definite matrix.")
            c, k, p, r = utils.decompose_cholesky(x[1:, 1:])
            
        pi = np.eye(self.d)
        pi[1:, 1:] = p
        xp = pi.dot(x).dot(pi.T)
        u = np.zeros(r+1)
        u[1:] = np.linalg.inv(c).dot(xp[0, 1:r+1])
        u[0] = xp[0, 0] - (u[1:]*u[1:]).sum()
        lst_proc_X = []
        
        for _ in range(num):
            W = utils.brownian(N=N, M=r, T=T)  # Of shape (N+1, r).
            cir_u0 = CIR(k=0, a=self.alpha-r, sigma=2, x0=u[0])  # The CIR generator.
            u0 = cir_u0(T=T, n=N, num=1)[0]  # Of shape (N+1,)
            proc_U = np.zeros((N+1, r+1))
            proc_U[:, 0] = u0
            proc_U[:, 1:] = (u[1:] + W).reshape(-1, 1)
            proc_X = self.hr(u=proc_U, r=r, c=c, k=k)  # Of shape (N+1, d, d)
            proc_X = [pi.T.dot(Xt).dot(pi) for Xt in proc_X]
            lst_proc_X.append(np.array(proc_X))

        return np.array(lst_proc_X)

    def wishart_i(self, T, n, N=100, num=1, x=None):
        '''
        :param T: Non-negative real number.
        :param n: Positive integer. The n of I_d^n.
        :param num: Positive integer. The number of discrete time points.
        :param x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :return: an np.array of shape (num, self.d, self.d).
        '''
        if x is None:
            x = self.x
        y = x.copy()
        I = np.eye(self.d)
        I[n:, n:] = np.zeros((self.d-n, self.d-n))

        for k in range(n):
            p = np.eye(self.d)
            p[0, 0] = p[k, k] = 0
            p[k, 0] = p[0, k] = 1
            if k == 0:
                Y = self.wishart_e(T, N=N, num=num, x=y)[:, -1]
                y = Y
            else:
                y = np.array([p.dot(y[i]).dot(p) for i in range(num)])
                Y = np.array([self.wishart_e(T, N=N, x=y[i])[0, -1] for i in range(num)])
                y = np.array([p.dot(Y[i]).dot(p) for i in range(num)])

        return y

    def to_integrate(self, a, b, s):
        '''
        Returns: terme to intergrate when s*b is inversible.
        '''
        D, V = utils.diag(s*b)
        exp = V.dot(np.exp(D)).dot(V.T)
        exp_T = V.T.dot(np.exp(D.T)).dot(V)

        return exp.dot(a.T).dot(a).dot(exp_T)

    def wishart(self, T, b, a, N=1, num=1, x=None, num_int=200):
        '''
        :param T: Non-negative real number.
        :param N: Positive integer. The number of discrete time points.
        :param x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :param b: Matrix of shape (self.d, self.d)
        :param a: Matrix of shape (self.d, self.d)
        :param num: num of simulations
        :return: an np.array of shape (num, self.d, self.d).
        '''
        if x is None:
            x = self.x
        # Here we shall find a method to calculate q.
        qT = utils.integrate(T, b, a, self.d, num_int=num_int)
        # Calculate p, cn, kn, 
        c, k, p, n = utils.decompose_cholesky(qT/T)
        # Build theta_t.
        theta = np.eye(self.d)
        theta[:n, :n] = c
        theta[n:, :n] = k
        theta = np.linalg.inv(p).dot(theta)
        m = scipy.linalg.expm(b*T)
        theta_inv = np.linalg.inv(theta)
        x_tmp = theta_inv.dot(m).dot(x).dot(m.T).dot(theta_inv.T)
        Y = self.wishart_i(T=T, n=n, N=N, num=num, x=x_tmp)
        X = np.array([theta.dot(Y[i]).dot(theta.T) for i in range(num)])
        
        return X

    def __call__(self, T, b, a, N=1, num=1, x=None):
        return self.wishart(T, b, a, N=N, x=x, num=num)

