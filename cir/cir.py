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
    
    @staticmethod
    def psi(k, t):
        '''
        Function psi_K(t).
        '''
        if k == 0:
            return t
        else:
            return (1-np.exp(-k*t))/k
    
#     max_q = 100
    def __init__(self, k, a, sigma, x0, exact=True):
        '''
        d Vt = (a - kVt)dt + sigma sqrt(Vt) d Wt
        * Params:
            k : Non-neg number.
            a : Pos number.
            sigma: Pos number.
            x0 : Initial value.
            exact ： boolean, indicating whether using excat or scheme simulation.
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
        
        self.exact = exact
        
        
#         if 'max_q' in kwargs:
#             max_q = kwargs['max_q']
#             assert isinstance(max_q, int) and max_q>0
#         else:
#             max_q = CIR.max_q
#         frc = Fraction.from_float(self.nu)
#         if not exact:
#             frc = frc.limit_denominator(max_q)
#         self.p = frc.numerator
#         self.q = frc.denominator
    

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
        assert x0 >= 0
        num = int(num)
        assert T>=0 and num>0
        assert n > 0
        h = T/n
        
        # If use the exact simulation.
        if self.exact:

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
        # If use the scheme simulation.
        else:
            sqrt_h = np.sqrt(h)
            V = np.zeros((num, n+1))
            V[:, 0] = x0
            
            # Case where sigma^2 <= 4a.
            if self.nu >= 1:
                N = np.random.normal(size=(num, n))
                for i in range(1, n+1):
                    V[:, i] = self.phi(V[:, i-1], h, sqrt_h*N[:, i-1])
            # Case where sigma^2 > 4a.
            else:
                Y = sampling.bounded_gauss(size=(num, n))
                U = np.random.uniform(size=(num, n))
                
                K2 = self.K2(h)
                for i in range(1, n+1):
                    # Indicating whether v_{t-1} >= K2 or v_{t-1} < K2
                    ind_out = V[:, i-1] >= K2
                    ind_in = ~ind_out
                # For x_{t-1} >= K2:
                    V[ind_out, i] = self.phi(V[ind_out, i-1], h, sqrt_h*Y[ind_out, i-1])
                # For x_{t-1} < K2:
                    V0_in = V[ind_in, i-1]
                    Vt_in = np.zeros_like(V0_in)
                    U_in = U[ind_in, i-1]
                    # 1st moment.
                    m1 = self.moment_1(V0_in, h) # Of shape same as V0_in.
                    # The partition pi.
                    pi = self.pi(V0_in, h)
                    ind_pos = U_in <= pi
                    ind_neg = ~ind_pos
                    # Calculate Vt_in.
                    Vt_in[ind_pos] = 1 / (2*pi[ind_pos])
                    Vt_in[ind_neg] = 1 / (2*(1-pi[ind_neg]))
                    Vt_in = Vt_in * m1
                    # Assign the value to V.
                    V[ind_in, i] = Vt_in
                    
            return V
                
    def scheme_fwd(self, x, t):
        assert x >= 0
        assert t >= 0
        
        # Case where sigma^2 <= 4a.
        if self.nu >= 1:
            N = np.random.normal(size=num)
            return self.phi(x, t, np.sqrt(t)*N)
        # Case where sigma^2 > 4a.
        else:
            # Check x0 and the bound.
            K2 = self.K2(t)
            # If x0 >= K2(t)
            if x0 >= K2:
                Y = sampling.bounded_gauss(size=num)
                return self.phi(x0, t, np.sqrt(t)*Y)
            # If x0 is out of the bound.
            else:
                m1 = self.moment_1(x0, t)
                pi = self.pi(x0, T)
                x_pos = m1 / (2*pi)
                x_neg = m1 / (2*(1-pi))

                U = np.random.rand(num)
                X = np.zeros_like(U)
                ind_pos = U<=pi
                ind_neg = ~ind_pos

                X[ind_pos] = x_pos
                X[ind_neg] = x_neg

                return X 
        

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
    
    def phi(self, x, t, w):
        '''
        Function phi(x, t, w) defined in Prop 2.1 of (ALFONSI, 2009).
        '''
        assert t >= 0
        k = self.k
        a = self.a
        sigma = self.sigma
        
        e_kt2 = np.exp(-k*t/2) # exp(-kt/2)
        tmp_apsi = (a - sigma*sigma/4) * CIR.psi(k, t/2)
        tmp_sqrt = tmp_apsi + e_kt2*x
        tmp_sqrt = np.sqrt(tmp_sqrt)
        tmp = tmp_sqrt + sigma/2 * w
        tmp = tmp * tmp
        
        return e_kt2 * tmp + tmp_apsi
    
    def K2(self, t):
        '''
        Function calculating the bound of x0 in case where sigma^2 > 4a.
        '''
        sigma = self.sigma
        a = self.a
        k = self.k
        
        exp_kt2 = np.exp(k*t/2)
        tmp_apsi = (sigma*sigma/4 - a) * CIR.psi(k, t/2)
        
        tmp_to_sqr = np.sqrt(exp_kt2*tmp_apsi) + np.sqrt(3*t) * sigma/2
        tmp = tmp_to_sqr * tmp_to_sqr
        tmp = tmp + tmp_apsi
        
        return exp_kt2 * tmp
        
    
    def moment_1(self, x, t):
        '''
        The 1st moment of X_t, i.e. E[X_t].
        '''
        k = self.k
        a = self.a
        
        return x*np.exp(-k*t) + a*CIR.psi(k, t)
    
    def moment_2(self, x, t):
        '''
        The 1st moment of X_t, i.e. E[X_t^2].
        '''
        k = self.k
        a = self.a
        sigma = self.sigma
        
        part_1 = self.moment_1(x, t)
        tmp_psi = CIR.psi(k, t)
        part_2 = sigma*sigma*tmp_psi
        part_2 = part_2 * (a*tmp_psi/2 + x*np.exp(-k*t))
        
        return part_1 + part_2
    
        
    def Delta(self, x, t):
        '''
        The discriminant, defined in (2.6) of (Alfonsi, 2009).
        '''
        m1 = self.moment_1(x, t)
        m2 = self.moment_2(x, t)
        
        return 1 - (m1*m1)/m2
        
    def pi(self, x, t):
        '''
        The function calculating pi(x, t), s.t.
        P[X_t = x_+] = pi(x, t).
        '''
        Delta = self.Delta(x, t)
        return (1 - np.sqrt(Delta)) / 2
    