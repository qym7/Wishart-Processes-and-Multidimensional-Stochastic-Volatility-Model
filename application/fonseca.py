import scipy.linalg
import numpy as np

from wishart import utils
from wishart import Wishart_e
import cir
from elgm import ELGM

class Fonseca_model:
    '''
    dY_t = (r - 1/2 * diag(X_t))dt + \sqrt(Xt)(\bar{\rho} dB_t + dW_t^T\rho),
    dX_t = (\alpha + bX_t + X_t^b^T)dt + \sqrt(X_t)dW_ta + a^T (dW_t)^T \sqrt(X_t).
    Where \alpha = \bar{\alpha} * a^Ta, for \bar{\alpha} > d-1.
    '''
    def __init__(self, r, rho, alpha, b, a):
        '''
        * Params:
            r, real-number. The interest rate.
            rho, np.array, of shape (d,). The correlation vector.
            alpha, np.array, of shape (d, d). 
            b, np.array, of shape (d, d).
            a, np.array, of shape (d, d). 
        '''
        d = len(rho)
        assert alpha.shape==(d, d) and b.shape==(d, d) and a.shape==(d, d)
        assert rho.shape == (d,)
        self.r = r
        self.d = d
        self.rho = rho
        self.bar_rho = np.sqrt(1 - np.linalg.norm(rho))
        self.alpha = alpha
        self.b = b
        self.a = a
        
        self.init_elgm()
        
    def init_elgm(self):
        '''
        This function calculates the parameters of elgm, and initialise the
        elgm instance.
        '''
        # Determine \bar{\alpha}
        aTa = (self.a.T) @ self.a
        tmp = self.alpha / aTa
        bar_alpha = tmp[0, 0]
        assert np.isclose(np.linalg.norm(tmp-bar_alpha), 0)
        self.bar_alpha = bar_alpha
        # Calculate I_d^n, u, b_u, and delta.
        c, k, p, n = utils.decompose_cholesky(aTa)
        self.n = n # n used in I_d^n.
        # Build uT.
        uT = np.eye(self.d)
        uT[:n, :n] = c
        uT[n:, :n] = k
        uT = np.matmul(np.linalg.inv(p), uT)
        self.u = uT.T # u, where x = u^T z u.
        self.inv_u = np.linalg.inv(self.u) # inverse of u.
        tmp_Idn = np.zeros(self.d)
        tmp_Idn[:n] = 1
        tmp_Idn = np.diag(tmp_Idn)
        delta = bar_alpha * tmp_Idn
        bu = np.matmul(self.inv_u.T, np.matmul(self.b, self.u.T)) # bu = (u^T)^-1 b u^T
        self.elgm_gen = ELGM(rho=self.rho, alpha=delta, b=bu, n=n)
        
    def step(self, x, y, dt, dBt=None, comb='r'):
        Xt = x
        Yt = y
        if comb=='r' or comb=='2' or comb==2:
            zeta = np.random.rand()
            if zeta < .5:
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt, dBt=dBt)
                Xt, Yt = self.step_L_tilde(x=Xt, y=Yt, dt=dt, comb=comb)
            else:
                Xt, Yt = self.step_L_tilde(x=Xt, y=Yt, dt=dt, comb=comb)
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt, dBt=dBt)
            return Xt, Yt
        elif comb=='1' or comb==1:
            if dBt is None:
                dBt = np.random.normal(size=(2, self.d)) * np.sqrt(dt/2)
            else:
                dBt = np.array(dBt)
                if dBt.shape != (2, self.d): # If dBt is of shape (d, d).
                    assert dBt.shape == (self.d,)
                    # Use the Brownian Bridge.
                    dBt_0 = dBt/2  + np.random.normal(size=(self.d)) * np.sqrt(dt/4)
                    dBt_1 = dBt - dBt_0
                    dBt = np.array([dBt_0, dBt_1])
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2, dBt = dBt[0])
            Xt, Yt = self.step_L_tilde(x=Xt, y=Yt, dt=dt, comb=comb)
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2, dBt = dBt[1])
            return Xt, Yt
                
    
    def step_L_1(self, x, y, dt, dBt=None):
        '''
        dYt = (r = diag(x)/2)dt + \bar{\rho}\sqrt(x)dBt.
        '''
        if dBt is None:
            dBt = np.random.norml(size=(self.d)) * np.sqrt(dt)
        c = utils.cholesky(x)
        Yt = (self.r - np.diag(x)/2)*dt + self.bar_rho * (c @ dBt)
        Xt = x
        return Xt, Yt
    
    def step_L_tilde(self, x, y, dt, comb='r'):
        r = np.matmul(self.inv_u.T, y)
        v = np.matmul(self.inv_u.T, np.matmul(x, self.inv_u))
        
        Vt, Rt = self.elgm_gen.step(x=r, v=y, dt=dt, comb=comb)
        Yt = np.matmul(self.u.T, Rt)
        Xt = np.matmul(self.u.T, np.matmul(Vt, self.u))
        return Xt, Yt
        