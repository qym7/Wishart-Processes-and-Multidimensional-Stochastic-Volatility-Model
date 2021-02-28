import numpy as np
import scipy.linalg

from wishart import utils
from wishart import Wishart
from wishart import Wishart_e
import cir

class ELGM:
    '''
    The Extention Linear Gaussian Model.
    Main ly used as a middle process for the simulation of Fonseca process.
    The SDEs are:
        dR_t = \sqrt(V_t) dW_t \rho,
        dV_t = (\alpha + bV_t + V_tb^T)dt + \sqrt(V_t)dW_tI_d^n + I_d^n dW_t^T \sqrt(V_t).
    '''
    def __init__(self, rho, alpha, b, n, epsilon=1):
        '''
        * Params:
            rho, np.array, of shape (d,). The correlation vector.
            alpha, np.array, of shape (d, d). 
            b, np.array, of shape (d, d).
            n, int. Indicating the matrix a as I_d^n.
            epsilon, normaly 1.
        '''
        d = len(rho)
        assert d == len(alpha) and d == len(b) and n <= d
        self.d = d
        self.rho = rho
        self.epsilon = epsilon # In normal case, epsilon is not used.
        self.alpha = alpha
        self.b = b
#         assert rho[n:].all() == 0 # Shall not have this assertion.
        self.n = n
        a = np.zeros(d)
        a[:n] = 1
        self.a = np.diag(a)
        
        # Justify whether alpha - dI_n^d is semi-pos-def.
        tar_mat = self.alpha - d * self.a
        W, V = np.linalg.eig(tar_mat) # Calculate the eig values.
        self.faster = (W >= 0).all()
        
        self.x_gen = Wishart_e(d=self.d, alpha=self.d-1)

#     def step_L_1(self, x, t, num_int=200):
#         return ex(x, t, self.b) + integrate(self.alpha, t, self.b, num_int)
    def step_L_1(self, x, dt):
        if self.faster:
            alpha = self.alpha - self.a
        else:
            alpha = self.alpha
        
        pass
        

    
    def step_L_c_q(self, x, y, dt, q):
        '''
        * return: Xt, Yt.
        '''
        assert x.shape == (self.d, self.d) and y.shape == (self.d,)
        assert q <= self.n
        epsilon_sqr = self.epsilon * self.epsilon
        rho_q = self.rho[q]
        
        Xt = self.x_gen.step(x=x, q=q, dt=epsilon_sqr*dt) # Generate Xt.
        dXt = Xt - x # Calculate dXt.
        Yt = y + rho_q / self.epsilon * dXt[q] # Calculate Yt.
        Yt[q] = y + rho_q / (2*self.epsilon) * (dXt[q, q] - epsilon_sqr * (d-1)*dt)
        return Xt, Yt
        
#     def step_L_c_q(self, x, y, t, q, keep_x=True):
#         assert x.shape == (self.d, self.d) and len(y) == self.d
#         assert q <= self.d

#         eqd = np.zeros((self.d, self.d))
#         eqd[q-1, q-1] = 1
#         x_generator = Wishart(x, self.d-1, 0, eqd) ## NEED MODIFY
#         xt = x_generator(T=t*self.epsilon*self.epsilon, x=x, N=1, num=1, method="exact")[0]
#         yt = np.zeros(self.d)
#         for i in range(self.d):
#             if i != q:
#                 yt[i] = y[i] + self.rho[i]/self.epsilon * (xt[q,i]-x[q,i])
#             else:
#                 yt[i] = y[i] + self.rho[i]/(2*self.epsilon) * (xt[q,i]-x[q,i]-self.epsilon*self.epsilon*(self.d-1)*t)
#         if keep_x:
#             return xt, yt
#         return yt

    def step_L_bar_q(self, u, y, dt, dWt, q):
        rho_q = self.rho[q]
        # Update Y.
        Yt = y + rho_q * np.matmul(u.T, dWt)
        tmp = np.sum(dWt[:,q] * dWt[:,q] - dt)
        Yt[q] = Yt[q] + self.epsilon * rho_q/2 * tmp
        # Update U.
        Ut = u
        Ut[:, q] = u[:, q] + self.epsilon * dWt[:, q]
        
        return Ut, Yt

    def step_L_bar(self, u, y, dt, dWt, comb='2'):
        '''
        '''
        Ut = u
        Yt = y
        if comb == '2' or comb == 2:
            Ut_1 = Ut.copy()
            Yt_1 = Yt.copy()
            Ut_2 = Ut.copy()
            Yt_2 = Yt.copy()
            for q in range(self.n):
                Ut_1, Yt_1 = self.step_L_bar_q(u=Ut_1, y=Yt_1, dt=dt, dWt=dWt, q=q)
                Ut_2, Yt_2 = self.step_L_bar_q(u=Ut_2, y=Yt_2, dt=dt, dWt=dWt, q=self.n-1-q)
            Ut = (Ut_1+Ut_2)/2 
            Yt = (Yt_1+Yt_2)/2
        else:
            pass
            
        return Ut, Yt
    
    def step_L_hat(self, x, y, dt, dWt):
        pass
    
# def ex(x, t, b):
#     return scipy.linalg.expm(t * b).dot(x).dot(scipy.linalg.expm(t * b.T))

# def integrate(alpha, T, b, num_int=200):
#     dt = T / num_int
#     lst_t = np.arange(num_int) * dt
#     dqt = np.array([dt*ex(alpha, t, b) for t in lst_t])

#     return dqt.cumsum()
def intgrl_etb(T, alpha, b, num_int=200):
    d = b.shape[0]
    assert b.shape == (d, d)
    assert alpha.shape == (d, d)
    dt = T / num_int
    lst_t = dt * np.arange(num_int)
    # Calculate e^tb.
    lst_etb = np.zeros((num_int, d, d))
    lst_etb[0] = np.eye(d)
    edtb = scipy.linalg.expm(dt*b) # exp(dt b).
    for i in range(1, num_int):
        lst_etb[i] = lst_etb[i-1] @ edtb
    
    lst_func_val = np.matmul(lst_etb, np.matmul(alpha, lst_etb.transpose(0, 2, 1)))
    intgrl_val = np.cumsum(lst_func_val, axis=0) * dt
    return intgrl_val
    
def get_order(n):
    binomial = np.random.binomial(size=n, n=1, p=.5)
    order = []
    for q in range(n):
        if binomial[q]:
            order = order + [q]
        else:
            order = [q] + order

    return order

