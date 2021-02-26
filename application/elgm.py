import numpy as np
import scipy

from wishart import utils
from wishart import Wishart
import cir

class ELGM:
    '''
    The Extention Linear Gaussian Model.
    Main ly used as a middle process for the simulation of Fonseca process.
    The SDEs are:
        dR_t = \sqrt(V_t) dW_t \rho,
        dV_t = (\alpha + bV_t + V_tb^T)dt + \sqrt(V_t)dW_tI_d^n + I_d^n dW_t^T \sqrt(V_t).
    '''
    def __init__(self, rho, alpha, b, n, epsilon):
        '''
        
        '''
        d = len(rho)
        assert d == len(alpha) and d == len(b) and n <= d
        self.d = d
        self.rho = rho
        self.epsilon = epsilon
        self.alpha = alpha
        self.b = b
        assert rho[n:].all() == 0
        self.n = n
        a = np.zeros(d)
        a[:n] = 1
        self.a = np.diag(a)
        
        # Justify whether alpha - dI_n^d is semi-pos-def.
        tar_mat = self.alpha - d * self.a
        W, V = np.linalg.eig(tar_mat) # Calculate the eig values.
        self.faster = (W >= 0).all()

    def step_L_1(self, x, t, num_int=200):
        return ex(x, t, self.b) + integrate(self.alpha, t, self.b, num_int)

    def step_L_c_q(self, x, y, t, q, keep_x=True):
        assert x.shape == (self.d, self.d) and len(y) == self.d
        assert q <= self.d

        eqd = np.zeros((self.d, self.d))
        eqd[q-1, q-1] = 1
        x_generator = Wishart(x, self.d-1, 0, eqd)
        xt = x_generator(T=t*self.epsilon*self.epsilon, x=x, N=1, num=1, method="exact")[0]
        yt = np.zeros(self.d)
        for i in range(self.d):
            if i != q:
                yt[i] = y[i] + self.rho[i]/self.epsilon * (xt[q,i]-x[q,i])
            else:
                yt[i] = y[i] + self.rho[i]/(2*self.epsilon) * (xt[q,i]-x[q,i]-self.epsilon*self.epsilon*(self.d-1)*t)

        if keep_x:
            return xt, yt
        return yt

    def step_L_c(self, x, y, t, keep_x=True):
        xt = x.copy()
        yt = y.copy()
        order = get_order(self.n)
        for q in order:
            xt, yt = self.step_L_c_q(x, y, t, q, keep_x=True)

        if keep_x:
            return xt, yt
        return yt

    def step_L_bar_q(self, u, y, w, t, q):
        assert len(w) == self.d
        ind = np.zeros((self.d, self.d))
        for i in range(self.n):
            ind[i,i] = 1
        ut = u.copy()
        ut[:, q] += self.epsilon*w
        yt = y + self.rho[q]*w.dot(u)
        yt[q] += self.epsilon*self.rho[q]/2*((w*w-t).sum())

        return ut, yt

    def step_L_bar(self, x, y, t, keep_x=True):
        ut = utils.decompose_cholesky(x)[0]
        yt = y.copy()
        w = utils.brownian_squrare(t, self.d, n_steps=1)[-1]
        order = get_order(self.n)
        for q in order:
            ut, yt = self.step_L_bar_q(ut, yt, w[:q], t, q)
        xt = ut.T.dot(ut)

        if keep_x:
            return xt, yt
        return yt


def ex(x, t, b):
    return scipy.linalg.expm(t * b).dot(x).dot(scipy.linalg.expm(t * b.T))

def integrate(alpha, T, b, num_int=200):
    dt = T / num_int
    lst_t = np.arange(num_int) * dt
    dqt = np.array([dt*ex(alpha, t, b) for t in lst_t])

    return dqt.cumsum()
    
def get_order(n):
    binomial = np.random.binomial(size=n, n=1, p=.5)
    order = []
    for q in range(n):
        if binomial[q]:
            order = order + [q]
        else:
            order = [q] + order

    return order

