import numpy as np
from scipy.integrate import quad

from wishart import linalg
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
            assert b.shape[0]==self.d and b.shape[1]==self.d
            self.b = b
        if a is None:
            self.a = np.eye(self.d)
        else:
            assert a.shape[0]==self.d and a.shape[1]==self.d
            self.a = a
        self.c, self.k, self.p, self.r = linalg.decompose_cholesky(self.x[1:, 1:])

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
        N = u.shape[0]
        for i in range(N):
            ut = u[i]
            m2 = np.zeros((self.d, self.d))
            m2[0, 0] = ut[0]+np.sum(ut[1:r + 1] * ut[1:r + 1])
            m2[0, 1:r + 1] = ut[1:r + 1]
            m2[1:r + 1, 0] = ut[1:r + 1]
            m2[1:r+1, 1:r+1] = np.eye(r)

            lst_hru.append(m1.dot(m2).dot(m3))
#         return m1.dot(m2).dot(m3)
        return np.array(lst_hru)

    def wishart_e(self, T, N, x=None):
        '''
        :param T: Non-negtive real number.
        :param N: Positive integer. The number of discretized time points.
        :x: Pos-def matrix. The initial value. If x is None, self.x is used.
        :return: an np.array of shape (N+1, self.d, self.d).
        '''
        assert T>=0 and N>0 
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
            c, k, p, r = linalg.decompose_cholesky(x[1:, 1:])
            
            
        pi = np.eye(self.d)
        pi[1:, 1:] = p
        xp = pi.dot(x).dot(pi.T)
        u = np.zeros(r+1)
        u[1:] = np.linalg.inv(c).dot(xp[0, 1:r+1])
        u[0] = xp[0,0] - (u[1:]*u[1:]).sum()
#         G = np.random.randn(r)
        W = linalg.brownian(N=N, M=r, T=T) # Of shape (N+1, r).

#         cir_u = CIR(0, self.alpha-self.r, 2, u[0], exact=False) # Do not know exact=false or true
        cir_u0 = CIR(k=0, a=self.alpha-r, sigma=2, x0=u[0]) # The CIR generator.
        u0 = cir_u0(T=T, n=N, num=1)[0] # Of shape (N+1,)
#         U_t = cir_u(t, 100, num=1)[0][-1]
        proc_U = np.zeros((N+1, r+1))
        proc_U[:, 0] = u0
        proc_U[:, 1:] = (u[1:] + W).reshape(-1, 1)
#         u[1:] = u[1:] + G*np.sqrt(t)
        proc_X = self.hr(u=proc_U, r=r, c=c, k=k) # Of shape (N+1, d, d)
        proc_X = [(pi.T).dot(Xt).dot(pi) for Xt in proc_X]
        return np.array(proc_X)

#         return (pi.T).dot(self.hr(u)).dot(pi)

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

        q = np.array([[quad(lambda s: np.exp(s*b).dot(a.T).dot(a).dot(np.exp(s*b.T))[i,j], 0, t)[0]
                       for i in range(self.d)] for j in range(self.d)])
        c, k, p, n = linalg.decompose_cholesky(q/t)
        theta = np.eye(self.d)
        theta[:n, :n] = c
        theta[n:, :n] = k
        theta = np.linalg.inv(p).dot(theta)
        m = np.exp(t*b)

        x = np.linalg.inv(theta).dot(m).dot(x).dot(m.T).dot(np.linalg.inv(theta).T)
        Y = self.wishart_i(t, x , n)

        return theta.dot(Y).dot(theta.T)

