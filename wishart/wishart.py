import numpy as np
from scipy.integrate import quad

from wishart import linalg
from cir import CIR

class Wishart():
    def __init__(self, x, alpha, b=0, a=None):
        '''
        :param x:
        :param alpha:
        :param b:
        :param a:
        '''
        try:
            np.linalg.cholesky(x)
        except:
            print("x is not a symetric positive matrix")
        self.x = x
        self.d = x.shape[0]
        assert alpha >= self.d - 1
        self.alpha = alpha
        self.b = b
        self.a = a or np.eye(self.d)
        self.c, self.k, self.p, self.r = linalg.decompose_cholesky(self.x[1:, 1:])

    def hr(self, u):
        '''
        :param u: an np.array of shape (self.r+1, 1)
        :return: an np.array of shape (self.d, self.d)
        '''
        m1 = np.eye(self.d)
        m1[1:self.r+1, 1:self.r+1] = self.c
        m1[self.r+1:self.d, 1:self.r+1] = self.k
        m2 = np.zeros((self.d, self.d))
        m2[0, 0] = u[0]+np.sum(u[1:self.r + 1] * u[1:self.r + 1])
        m2[0, 1:self.r + 1] = u[1:self.r + 1]
        m2[1:self.r + 1, 0] = u[1:self.r + 1]
        m2[1:self.r+1, 1:self.r+1] = np.eye(self.r)
        m3 = m1.copy().T

        return m1.dot(m2).dot(m3)

    def wishart_e(self, t, x):
        '''
        :param t:
        :return: an np.array of shape (self.d, self.d)
        '''
        assert t>0
        pi = np.eye(self.d)
        pi[1:, 1:] = self.p
        xp = pi.dot(x).dot(pi.T)
        u = np.zeros(self.r+1)
        u[1:] = np.linalg.inv(self.c).dot(xp[1])
        u[0] = xp[0,0] - (u[1:]*u[1:]).sum()
        G = np.random.randn(self.r)

        cir_u = CIR(0, self.alpha-self.r, 2, u[0], exact=False) # Do not know exact=false or true
        U_t = cir_u(t, 100, num=1)[0][-1]

        u[0] = U_t
        u[1:] = u[1:] + G*np.sqrt(t)

        return (pi.T).dot(self.hr(u)).dot(pi)

    def wishart_i(self, t, x, n):
        y = x.copy()
        I = np.eye(self.d)
        I[n:, n:] = np.zeros((self.d-n, self.d-n))
        for k in range(n):
            p = np.eye(self.d)
            p[0,0] = p[k,k] = 0
            p[k,0] = p[0,k] = 1
            Y = self.wishart_e(t, p.dot(y).dot(p))
            y = p.dot(Y).dot(p.T)

        return y

    def __call__(self, x, t, b, a, num=1):
        assert b.shape == (self.d, self.d) and a.shape == (self.d, self.d)

        q = quad(lambda s: np.exp(s*b).dot(a.T).dot(a).dot(np.exp(s*b.T)), 0, t)
        c, k, p, n = linalg.decompose_cholesky(q/t)
        theta = np.eye(self.d)
        theta[:n, :n] = c
        theta[n:, :n] = k
        theta = np.linalg.inv(p).dot(theta)
        m = np.exp(t*b)

        x = np.linalg.inv(theta).dot(m).dot(x).dot(m.T).dot(np.linalg.inv(theta).T)
        Y = self.wishart_i(t, x , n)

        return theta.dot(Y).dot(theta.T)

