'''
Defined the class CIR which generates discretised CIR process.
20 Oct. 2020.
Benxin ZHONG
'''
import numpy as np
import sampling

def psi(k, t):
    '''
    Function psi_K(t).
    '''
    if k == 0:
        return t
    else:
        return (1 - np.exp(-k * t)) / k

class CIR:
    '''
    The Class of CIR process generator.
    '''
    def __init__(self, k, a, sigma, x0):
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
        if x0 < 0:
            print(f'x0 is {x0}.')
        assert x0 >= 0
        self.x0 = x0
        self.nu = 4*a / self.sigmasqr
        
    def __call__(self, T, n, num=1, x0=None, method="exact"):
        '''
        Function used to generate the discretized CIR process.
        * Params:
            T : Non-negative real number.
            n : Positive integer. The number of discretized time points. 
            num : Positive integer. The number of independent CIR processes to generate.
            x0 : Real number. The initial value of V. If x0 is None, self.x0 is used.
            method: String. It indicates the simulation method:
                    - "exact": exact method
                    - "2": scheme of order 2
                    - "3": scheme of order 3
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

        if method == "exact":
            return self.exact_cir(h, n, num=num, x0=x0)
        elif method == "2":
            return self.scheme_2_cir(h, n, num=num, x0=x0)
        elif method == "3":
            return self.scheme_3_cir(h, n, num=num, x0=x0)

    def exact_cir(self, t, n, num, x0):
        nita = self.nita(t)
        factor = np.exp(-1 * self.k * t) / nita
        # Generate Vt.
        V = np.zeros((num, n + 1))
        V[:, 0] = x0
        for i in range(1, n + 1):
            Vt = V[:, i - 1]
            lam_t = Vt * nita
            # Generate the chi-square distribution.
            Vt1 = np.random.noncentral_chisquare(df=self.nu, nonc=lam_t)
            Vt1 = Vt1 * factor  # Calculate V_t_{i+1}.
            V[:, i] = Vt1

        return V

    def scheme_2_cir(self, t, n, num, x0):
        sqrt_h = np.sqrt(t)
        V = np.zeros((num, n + 1))
        V[:, 0] = x0

        if self.nu >= 1:
            # Case where sigma^2 <= 4a.
            N = np.random.normal(size=(num, n))
            for i in range(1, n + 1):
                V[:, i] = self.phi(V[:, i - 1], t, sqrt_h * N[:, i - 1])
        else:
            # Case where sigma^2 > 4a.
            Y = sampling.bounded_gauss(size=(num, n))
            U = np.random.uniform(size=(num, n))
            K2 = self.K2(t)
            for i in range(1, n + 1):
                # Indicating whether v_{t-1} >= K2 or v_{t-1} < K2
                ind_out = V[:, i - 1] >= K2
                ind_in = ~ind_out
                # For x_{t-1} >= K2:
                V[ind_out, i] = self.phi(V[ind_out, i - 1], t, sqrt_h * Y[ind_out, i - 1])
                # For x_{t-1} < K2:
                V0_in = V[ind_in, i - 1]
                Vt_in = np.zeros_like(V0_in)
                U_in = U[ind_in, i - 1]
                # 1st moment.
                m1 = self.moment_1(V0_in, t)  # Of shape same as V0_in.
                # The partition pi.
                pi = self.pi(V0_in, t)
                ind_pos = U_in <= pi
                ind_neg = ~ind_pos
                # Calculate Vt_in.
                Vt_in[ind_pos] = 1 / (2 * pi[ind_pos])
                Vt_in[ind_neg] = 1 / (2 * (1 - pi[ind_neg]))
                Vt_in = Vt_in * m1
                # Assign the value to V.
                V[ind_in, i] = Vt_in

        return V

    def scheme_3_cir(self, t, n, num, x0):
        psi_kt = psi(-1*self.k, t)
        Y = sampling.bounded_gauss(size=(num, n), order=3)
        epsilon = np.random.randint(2, size=(num, n)) * 2 - 1
        zeta = np.random.randint(3, size=(num, n))
        U = np.random.uniform(size=(num, n))

        V = np.zeros((num, n + 1))
        V[:, 0] = x0

        K3 = self.K3(t)
        for i in range(1, n + 1):
            # Seperate x.
            ind_out = V[:, i - 1] >= K3
            ind_in = ~ind_out
            # Process x >= K3.
            zeta_out = zeta[ind_out, i - 1]  # Random variable ksi.
            Y_out = Y[ind_out, i - 1]
            epsilon_out = epsilon[ind_out, i - 1]
            x0 = V[ind_out, i - 1]
            x1 = self.hat_x(x0, psi_kt, epsilon_out, zeta_out, Y_out)  # \hat{X}_{\psi_{-k}(t)}^{x, k=0}.
            x1 = np.exp(-self.k * t) * x1
            V[ind_out, i] = x1
            # Process x < K3.
            U_in = U[ind_in, i - 1]
            x0 = V[ind_in, i - 1]
            x1 = np.zeros_like(x0)
            m1, m2, m3 = self.moment_3(x0, t)
            tmp = m2 - m1 * m1
            s = (m3 - m1 * m2) / tmp
            p = (m1 * m3 - m2 * m2) / tmp
            
            sqrt_Delta = np.sqrt(s * s - 4 * p)
            x_pos = (s + sqrt_Delta) / 2
            x_neg = (s - sqrt_Delta) / 2
            pi = (m1 - x_neg) / (sqrt_Delta)
            ind_pos = U_in <= pi
            ind_neg = ~ind_pos
            x1[ind_pos] = x_pos[ind_pos]
            x1[ind_neg] = x_neg[ind_neg]
            V[ind_in, i] = x1

        return V
                
    def hat_x(self, x, t, epsilon, zeta, Y):
        '''
        Function used to calculate \hat{X}_t^{x, k=0}.
        '''
        shape_x = x.shape
        assert epsilon.shape == shape_x and zeta.shape == shape_x and Y.shape == shape_x
        sqrt_t = np.sqrt(t)
        xt = np.zeros_like(x)
        
        ind_0 = zeta==0
        ind_1 = zeta==1
        ind_2 = zeta==2
        if self.nu >= 1: # sigma^2 <= 4a.
            # zeta = 0.
            tmp = self.cir_x1(t=sqrt_t*Y[ind_0], x=x[ind_0])
            tmp = self.cir_x0(t=t, x=tmp, k=0)
            xt[ind_0] = self.cur_x(t=epsilon[ind_0]*t, x=tmp)
            # zeta = 1.
            tmp = self.cir_x1(t=sqrt_t*Y[ind_1], x=x[ind_1])
            tmp = self.cur_x(t=epsilon[ind_1]*t, x=tmp)
            xt[ind_1] = self.cir_x0(t=t, x=tmp, k=0)
            # zeta = 2.
            tmp = self.cur_x(t=epsilon[ind_2]*t, x=x[ind_2])
            tmp = self.cir_x1(t=sqrt_t*Y[ind_2], x=tmp)
            xt[ind_2] = self.cir_x0(t=t, x=tmp, k=0)
            
        else: # sigma^2 > 4a.
            # zeta = 0.
            tmp = self.cir_x0(t=t, x=x[ind_0], k=0)
            tmp = self.cir_x1(t=sqrt_t*Y[ind_0], x=tmp)
            xt[ind_0] = self.cur_x(t=epsilon[ind_0]*t, x=tmp)
            # zeta = 1.
            tmp = self.cir_x0(t=t, x=x[ind_1], k=0)
            tmp = self.cur_x(t=epsilon[ind_1]*t, x=tmp)
            xt[ind_1] = self.cir_x1(t=sqrt_t*Y[ind_1], x=tmp)
            # zeta = 2.
            tmp = self.cur_x(t=epsilon[ind_2]*t, x=x[ind_2])
            tmp = self.cir_x0(t=t, x=tmp, k=0)
            xt[ind_2] = self.cir_x1(t=sqrt_t*Y[ind_2], x=tmp)
        
        return xt

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
    
    def cir_x0(self, x, t, k):
        a = self.a
        sigma = self.sigma
        psi_k = psi(k, t)
        tmp = (a - sigma*sigma/4) * psi_k

        return x * np.exp(-k*t) + tmp
    
    def cir_x1(self, x, t):
        sigma = self.sigma
        tmp = np.sqrt(x) + sigma*t/2
        tmp = tmp.clip(min=0)
        
        return tmp * tmp
    
    def cur_x(self, x, t):
        sigma = self.sigma
        a = self.a
        tmp = np.abs(a - sigma*sigma/4)
        
        return x + t*sigma*np.sqrt(tmp)/np.sqrt(2)
    
    def phi(self, x, t, w):
        '''
        Function phi(x, t, w) defined in Prop 2.1 of (ALFONSI, 2009).
        '''
        assert t >= 0
        k = self.k
        a = self.a
        sigma = self.sigma
        
        e_kt2 = np.exp(-k*t/2) # exp(-kt/2)
        tmp_apsi = (a - sigma*sigma/4) * psi(k, t/2)
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
        tmp_apsi = (sigma*sigma/4 - a) * psi(k, t/2)
        
        tmp_to_sqr = np.sqrt(exp_kt2*tmp_apsi) + np.sqrt(3*t) * sigma/2
        tmp = tmp_to_sqr * tmp_to_sqr
        tmp = tmp + tmp_apsi
        
        return exp_kt2 * tmp
    
    def K3(self, t):
        a = self.a
        sigma = self.sigma
        k = self.k
        _psi = psi(-k, t)
        if self.nu < 1: # 4a < sigma^2.
            tmp = np.sqrt(sigma*sigma/4 - a) # \sqrt{sigma^2/4 - a}
            tmp = np.sqrt(sigma / (np.sqrt(2)) * tmp)
            tmp = tmp + sigma/2 * np.sqrt(3 + np.sqrt(6))
            tmp = tmp*tmp + sigma*sigma/4 - a
            return _psi * tmp

        elif self.nu < 3: #4a/3 < sigma^2 < 4a.
            tmp = np.sqrt(a - sigma*sigma/4)
            tmp = sigma * tmp / np.sqrt(2)
            tmp = np.sqrt(sigma*sigma/4 - a + tmp)
            tmp = tmp + sigma/2 * np.sqrt(3 + np.sqrt(6))
            tmp = tmp*tmp
            return _psi * tmp

        else: # sigma^2 < 4a/3.
            tmp = np.sqrt(a - sigma*sigma/4)
            tmp = sigma * tmp / np.sqrt(2)
            return _psi * tmp

    def moment_1(self, x, t):
        '''
        The 1st moment of X_t, i.e. E[X_t].
        '''
        k = self.k
        a = self.a
        
        return x*np.exp(-k*t) + a*psi(k, t)
    
    def moment_2(self, x, t):
        '''
        The 1st moment of X_t, i.e. E[X_t^2].
        return: (m1, m2).
        '''
        k = self.k
        a = self.a
        sigma = self.sigma
        
        m1 = self.moment_1(x, t)
        tmp_psi = psi(k, t)
        part_2 = sigma*sigma*tmp_psi
        part_2 = part_2 * (a*tmp_psi/2 + x*np.exp(-k*t))
        m2 = m1*m1 + part_2
        
        return m1, m2
    
    def moment_3(self, x, t):
        k = self.k
        a = self.a
        sigma = self.sigma
        
        m1, m2 = self.moment_2(x, t)
        psi_k = psi(k, t)
        tmp = 3*x*np.exp(-k*t) + a * psi_k
        tmp = psi_k * (a+sigma*sigma/2) * tmp
        tmp = 2*x*x*np.exp(-2*k*t) + tmp
        tmp = sigma*sigma * psi_k * tmp
        m3 = m1*m2 + tmp
        
        return m1, m2, m3

    def Delta(self, x, t):
        '''
        The discriminant, defined in (2.6) of (Alfonsi, 2009).
        '''
        m1, m2 = self.moment_2(x, t)
        return 1 - (m1*m1)/m2
        
    def pi(self, x, t):
        '''
        The function calculating pi(x, t), s.t.
        P[X_t = x_+] = pi(x, t).
        '''
        Delta = self.Delta(x, t)
        return (1 - np.sqrt(Delta)) / 2
    