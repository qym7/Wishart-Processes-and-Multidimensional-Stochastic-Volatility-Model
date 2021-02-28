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
        dY_t = \sqrt(X_t) dW_t \rho,
        dX_t = (\alpha + bX_t + X_tb^T)dt + \epsilon (\sqrt(X_t)dW_tI_d^n + I_d^n dW_t^T \sqrt(X_t)).
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

    def gen(self, x, y, T, N=1, num=1, comb='r', **kwargs):
        assert x.shape == (self.d, self.d)
        assert y.shape == (self.d,)
        dt = T/N
        lst_t = np.arange(N+1) * dt
        
        # Calculate and store L1.
        if self.faster:
            alpha = self.alpha - self.a
        else:
            alpha = self.alpha
        if comb=='r' or comb=='2' or comb==2:
            t = dt
        elif comb=='1' or comb==1:
            t = dt/2
        if 'num_int' in kwargs:
            num_int = kwargs['num_int']
        else:
            num_int = 200
        self.cal_tmp(t=t, alpha=alpha, b=b, num_int=num_int)
        
        # check dWt or Wt.
#         if 'dWt' in kwargs:
#             dWt = kwargs['dWt']
#             assert dWt.shape==(N, self.d, self.d) or dWt.shape==(N, 2, self.d, self.d)
#         elif 'Wt' in kwargs:
#             Wt = kwargs['Wt']
#             assert Wt.shape==(N+1, self.d, self.d) or Wt.shape==(2*N+1, self.d, self.d)
#             dWt = Wt[1:] - Wt[:-1]
#             if len(dWt) == 2*N:
#                 dWt = dWt.reshape(N, 2, self.d, self.d)
#         else:
#             dWt = None
        dWt = None
            
        # Generate.
        lst_trace_Xt = np.zeros((num, N+1, self.d, self.d))
        lst_trace_Yt = np.zeros((num, N+1, self.d))
        lst_trace_Xt[:, 0] = x
        lst_trace_Yt[:, 0] = y
        for i in range(num):
             for j in range(1, N+1):
                    Xt, Yt = self.step(x=lst_trace_Xt[i, j-1], y=lst_trace_Yt[i, j-1], dt=dt, dWt=dWt, comb=comb)
                    lst_trace_Xt[i, j] = Xt
                    lst_trace_Yt[i, j] = Yt
        if 'trace' in kwargs and not kwargs['trace']:
            return lst_trace_Xt[:, -1], lst_trace_Yt[:, -1]
        else:
            return lst_trace_xt, lst_trace_Yt
    
    def step(self, x, y, dt, dWt=None, comb='r'):
        if self.faster:
            Xt, Yt = self.step_fast(x=x, y=y, dt=dt, dWt=dWt, comb=comb)
        else:
            Xt, Yt = self.step_no_fast(x=x, y=y, dt=dt, comb=comb)
            
        return Xt, Yt
    
    def step_fast(self, x, y, dt, dWt=None, comb='r'):
        if dWt is None:
            dWt = np.random.normal(size=(self.d, self.d)) * np.sqrt(dt)
            
        Xt = x
        Yt = y
        if comb=='r' or comb=='2' or comb==2:
            zeta = np.random.rand()
            if zeta < .5:
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt)
                Xt, Yt = self.step_L_hat(x=Xt, y=Yt, dt=dt, dWt=dWt, comb='r')
            else:
                Xt, Yt = self.step_L_hat(x=Xt, y=Yt, dt=dt, dWt=dWt, comb='r')
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt)
            return Xt, Yt
        elif comb=='1' or comb==1:
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2)
            Xt, Yt = self.step_L_hat(x=Xt, y=Yt, dt=dt, dWt=dWt, comb='1')
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2)
            return Xt, Yt
    
    def step_no_fast(self, x, y, dt, comb='r'):
        Xt = x
        Yt = y
        if comb=='r' or comb=='2' or comb==2:
            zeta = np.random.rand()
            if zeta < .5:
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt)
                Xt, Yt = self.step_L_c(x=Xt, y=Yt, dt=dt, comb='r')
            else:
                Xt, Yt = self.step_L_c(x=Xt, y=Yt, dt=dt, comb='r')
                Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt)
            return Xt, Yt
        elif comb=='1' or comb==1:
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2)
            Xt, Yt = self.step_L_c(x=Xt, y=Yt, dt=dt, comb='1')
            Xt, Yt = self.step_L_1(x=Xt, y=Yt, dt=dt/2)
            return Xt, Yt
            
            
        
    
    def step_L_1(self, x, y, dt):
        '''
        In order to reduce the repeated calculation, the function 
        uses the pre-stored value of \int_{0}^{t}e^(sb)(\alpha) ds, 
        and the value of e^(tb), which are stored in the tuple 
        `self.tmp_intgrl`.
        Note that the \alpha value is already considered when pre-
        calculating, therefore this function could ignore the 
        `self.faster` flag. Remark that when using combination method
        1, the L_1 step size is acctually dt/2. And this shall be con-
        sidered also in the pre-calculation. 
        '''
#         if self.faster:
#             alpha = self.alpha - self.a
#         else:
#             alpha = self.alpha
        t, tmp_etb, tmp_int_etb = self.tmp_intgrl
        assert t == dt
        # Xt = e^(tb) (x) + \int_{0}^{t} e^(sb)(\alpha) ds
        # e^(tb) (x) = e^(tb) x (e^tb)^T.
        Xt = np.matmul(tmp_etb, np.matmul(x, tmp_etb.T)) + tmp_int_etb
        Yt = y
        return Xt, Yt
    
    def step_L_hat(self, x, y, dt, dWt, comb='r'):
        c = utils.cholesky(x)
        Ut, Yt = self.step_L_bar(u=c, y=y, dt=dt, dWt=dWt, comb=comb)
        Xt = (Ut.T) @ Ut
        
        return Xt, Yt
    
    def step_L_c(self, x, y, dt, comb='r'):
        Xt = x
        Yt = y
        if comb == 'r' or comb == '2' or comb == 2:
            zeta = np.random.rand(self.n-1)
            seq_q = [0]
            for q in range(1, self.n):
                if zeta[q-1] < .5:
                    seq_q.append(q) # 96.6ns.
                else:
                    seq_q = [q] + seq_q # 138ns.
        
            for q in seq_q:
                Xt, Yt = self.step_L_c_q(x=Xt, y=Yt, dt=dt, q=q)
        elif comb == '1' or 1:
            for q in range(1, self.n)[::-1]:
                Xt, Yt = self.step_L_c_q(x=Xt, y=Yt, dt=dt/2, q=q)
            Xt, Yt = self.step_L_c_q(x=Xt, y=Yt, dt=dt, q=0)
            for q in range(1, self.n):
                Xt, Yt = self.step_L_c_q(x=Xt, y=Yt, dt=dt/2, q=q)
        return Xt, Yt

    
    def step_L_c_q(self, x, y, dt, q):
        '''
        * return: Xt, Yt.
        '''
#         assert x.shape == (self.d, self.d) and y.shape == (self.d,)
#         assert q <= self.n
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
        

    def step_L_bar(self, u, y, dt, dWt=None, comb='r', zeta=None):
        '''
        * Params:
            comb :  '1'/1 or 'r'/'2'/2. Default is 'r'. Indicating the combination
                    method of the generators. If '1'/1 is specified, `dWt` must be
                    `None` or of shape (2, d, d), each indicating the dWt with step
                    size dt/2, or of shape (d, d). Then the dWt with precision dt/2
                    is generated using Brownian bridge.
            zeta : if 'r' is specified in comb, zeta is the random variable deter-
                    ming the direction of combination.
        '''
        Ut = u
        Yt = y
        if comb == '1' or comb == 1:
            if dWt is None:
                dWt = np.random.normal(size=(2, self.d, self.d)) * np.sqrt(dt/2)
            else:
                dWt = np.array(dWt)
                if dWt.shape != (2, self.d, self.d):
                    assert dWt.shape == (self.d, self.d)
                    dWt_0 = dWt/2  + np.random.normal(size=(self.d, self.d)) * np.sqrt(dt/4)
                    dWt_1 = dWt - dWt_0
                    dWt = np.array([dWt_0, dWt_1])
            for q in range(1, self.n)[::-1]:
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt/2, dWt=dWt[0], q=q)
            Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt/2, dWt=dWt[0]+dWt[1], q=0)
            for q in range(1, self.n):
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt/2, dWt=dWt[1], q=q)
            return Ut, Yt
#             Ut_1 = Ut.copy()
#             Yt_1 = Yt.copy()
#             Ut_2 = Ut.copy()
#             Yt_2 = Yt.copy()
#             for q in range(self.n):
#                 Ut_1, Yt_1 = self.step_L_bar_q(u=Ut_1, y=Yt_1, dt=dt, dWt=dWt, q=q)
#                 Ut_2, Yt_2 = self.step_L_bar_q(u=Ut_2, y=Yt_2, dt=dt, dWt=dWt, q=self.n-1-q)
#             Ut = (Ut_1+Ut_2)/2 
#             Yt = (Yt_1+Yt_2)/2
        elif comb == 'r' or comb == '2' or comb == 2:
            if zeta is None:
                zeta = np.random.rand(self.n-1)
            else:
                assert len(zeta) == (self.n-1)
            
            # Construct the combination order.
            seq_q = [0]
            for q in range(1, self.n):
                if zeta[q-1] < .5:
                    seq_q.append(q) # 96.6ns.
                else:
                    seq_q = [q] + seq_q # 138ns.
            
            for q in seq_q:
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt, dWt=dWt, q=q)
#             if zeta < .5:
#                 for q in range(self.n):
#                     Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt, dWt=dWt, q=q)
#             else:
#                 for q in range(self.n):
#                     Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt, dWt=dWt, q=self.n-1-q)
            return Ut, Yt
        else:
            pass
    
    
    def cal_tmp(self, t, alpha, b, num_int=200):
        tmp_etb = scipy.linalg.expm(t*b)
        tmp_int_etb = intgrl_etb(T=t, alpha=alpha, b=b, num_int=num_int)[-1]
        self.tmp_intgrl = [t, tmp_etb, tmp_int_etb]
            

    
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
    
    