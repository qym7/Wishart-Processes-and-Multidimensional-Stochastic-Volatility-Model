'''
Defined the class CIR which generates discretised CIR process.
20 Oct. 2020.
Benxin ZHONG
'''
import numpy as np
from fractions import Fraction
import sampling

class CIR:
    '''
    The Class of CIR process generator.
    '''
    max_q = 100
    def __init__(self, k, a, sigma, x0, exact=False, **kwargs):
        '''
        d Vt = (a - kVt)dt + sigma sqrt(Vt) d Wt
        * Params:
            k : Non-neg number.
            a : Pos number.
            sigma: Pos number.
            x0 : Initial value.
        '''
        assert k >= 0
        assert a > 0
        assert sigma > 0
        self.k = k
        self.a = a
        self.sigma = sigma
        self.sigmasqr = sigma*sigma
        self.x0 = x0

        self.nu = 4*a / self.sigmasqr
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
        '''
        Function nita, nita(h) = 4k exp(-kh) / (sigma^2(1-exp(-kh))).
        '''
        if self.k == 0:
            return 4/(self.sigmasqr*h)

        tmp = np.exp(-1*self.k * h)
        nomerator = 4 * self.k * tmp
        denominator = self.sigmasqr * (1-tmp)
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
        factor = np.exp(-1*self.k*h) / nita

        # Generate Vt.
        V = np.zeros((num, n+1))
        V[:, 0] = x0
        for i in range(1, n+1):
            Vt = V[:, i-1]
            lam_t = Vt * nita
            # Generate the chi-square distribution.
#             Vt1 = sampling.chi_2(self.p, self.q, lam=lam_t)
            Vt1 = np.random.noncentral_chisquare(df = self.nu, nonc=lam_t)

            Vt1 = Vt1 * factor # Calculate V_t_{i+1}.
            V[:, i] = Vt1
        
        return V