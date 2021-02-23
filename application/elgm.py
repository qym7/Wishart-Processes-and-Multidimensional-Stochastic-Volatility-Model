import numpy as np
from wishart import utils
import cir

class ELGM:
    '''
    The Extention Linear Gaussian Model.
    Main ly used as a middle process for the simulation of Fonseca process.
    The SDEs are:
        dR_t = \sqrt(V_t) dW_t \rho,
        dV_t = (\alpha + bV_t + V_tb^T)dt + \sqrt(V_t)dW_tI_d^n + I_d^n dW_t^T \sqrt(V_t).
    '''
    def __init__(self, rho, alpha, b, n):
        '''
        
        '''
        d = len(rho)
        assert d == len(alpha) and d == len(b) and n <= d
        self.d = d
        self.rho = rho
        self.alpha = alpha
        self.b = b
        self.n = n
        a = np.zeros(d)
        a[:n] = 1
        self.a = np.diag(a)
        
        # Justify whether alpha - dI_n^d is semi-pos-def.
        tar_mat = self.alpha - d * self.a
        W, V = np.linalg.eig(tar_mat) # Calculate the eig valus.
        self.faster = (W >= 0).all()
        
        
        def step_L_1(self, );
            pass
        
        def step_L_c(self, ):
            pass
        
        def step_L_bar(self, ):
            pass
        
        
        
    
    