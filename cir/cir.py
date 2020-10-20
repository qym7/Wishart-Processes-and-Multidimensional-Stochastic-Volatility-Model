import numpy as np
from fractions import Fraction
import random_CIR

class CIR:
    '''
    The Class of CIR process generator.
    '''
    max_q = 100
    def __init__(self, kappa, theta, epsilon, x0, exact=False, **kwargs):
        assert kappa > 0
        assert theta > 0
        assert epsilon > 0
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.x0 = x0
        
        self.nu = 4*kappa*theta / (epsilon*epsilon)
        if 'max_q' in kwargs:
            max_q = kwargs['max_q']
            assert isinstance(max_q, int) and max_q>0
        else:
            max_q = CIR.max_q
        frc = Fraction.from_float(self.nu)
        if not exact:
            frc = frc.limit_denominator(max_q)
        self.p = frc.numerator
        self.q = frc.denominator
    
    def nita(self, h):
        tmp = np.exp(-1*self.kappa * h)
        nomerator = 4 * self.kappa * tmp
        denominator = self.epsilon*self.epsilon * (1-tmp)
        return nomerator/denominator
    
    def __call__(self, T, n, num=1, x0=None):
        '''
        Function used to generate the discretized CIR process.
        * Params:
            T : Non-negative real number.
            n : Positive integer. The number of discretized time points.
            num : Positive integer. The number of independent CIR processes to generate.
            x0 : Real number. The initial value of V. If x0 is None, self.x0 is used.
        * Return:
            V : The generated process. Of shape (num, n+1).
        '''
        if x0 is None:
            x0 = self.x0
        num = int(num)
        assert T>=0 and n>0 and num>0
        h = T/n
        
        nita = self.nita(h)
        factor = np.exp(-1*self.kappa*h) / nita
        
        # Generate Vt.
        V = np.zeros((num, n+1))
        V[:, 0] = x0
        for i in range(1, n+1):
            Vt = V[:, i-1]
            lam_t = Vt * nita
            # Generate the chi-square distribution.
            Vt1 = random_CIR.chi_2(self.p, self.q, lam=lam_t)
            Vt1 = Vt1 * factor # Calculate V_t_{i+1}.
            V[:, i] = Vt1
        
        return V