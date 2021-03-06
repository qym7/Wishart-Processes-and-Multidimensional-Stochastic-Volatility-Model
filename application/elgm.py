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
        self.n = n
        a = np.zeros(d)
        a[:n] = 1
        self.a = np.diag(a)
        
        # Justify whether alpha - dI_n^d is semi-pos-def.
        tar_mat = self.alpha - d * self.a
        W, V = np.linalg.eig(tar_mat) # Calculate the eig values.
        self.faster = (W >= 0).all()
        
        self.x_gen = Wishart_e(d=self.d, alpha=self.d-1)

    def gen(self, x, y, T, N=1, num=1, comb='r', **kwargs):
        assert x.shape == (self.d, self.d)
        assert y.shape == (self.d,)
        dt = T/N
        self.pre_gen(T=T, N=N, num=num, comb=comb, **kwargs)
        dWt = None

        # Generate.
        lst_trace_Xt = np.zeros((num, N+1, self.d, self.d))
        lst_trace_Yt = np.zeros((num, N+1, self.d))
        lst_trace_Xt[:, 0] = x
        lst_trace_Yt[:, 0] = y
        
        lst_it = range(num)
        if 'tqdm' in kwargs:
            tqdm = kwargs['tqdm']
            lst_it = tqdm(lst_it)
        for i in lst_it:
             for j in range(1, N+1):
                    Xt, Yt = self.step(x=lst_trace_Xt[i, j-1], y=lst_trace_Yt[i, j-1], dt=dt, dWt=dWt, comb=comb)
                    lst_trace_Xt[i, j] = Xt
                    lst_trace_Yt[i, j] = Yt
        if 'trace' in kwargs and not kwargs['trace']:
            return lst_trace_Xt[:, -1], lst_trace_Yt[:, -1]
        else:
            return lst_trace_Xt, lst_trace_Yt

    def pre_gen(self, T, N=1, comb='r', **kwargs):
        '''
        Pre process for generating. The function determines the alpha and b matrices
        used in the L1 generator, and calculate and store the tmp values.
        This function can be called individually apart from the function `gen`.
        '''
        dt = T/N
        
        # Calculate and store L1.
        if self.faster:
            alpha = self.alpha - self.a
        else:
            alpha = self.alpha
        if comb=='r' or comb=='2' or comb==2 or comb=="euler":
            dt = dt
        elif comb=='1' or comb==1:
            dt = dt/2
        if 'num_int' in kwargs:
            num_int = kwargs['num_int']
        else:
            num_int = 200
        self.cal_tmp(t=dt, alpha=alpha, b=self.b, num_int=num_int)
    
    def step(self, x, y, dt, dWt=None, comb='r'):
        if comb=='euler':
            Xt, Yt = self.step_euler(x=x, y=y, dt=dt, dWt=dWt)
        elif self.faster:
            Xt, Yt = self.step_fast(x=x, y=y, dt=dt, dWt=dWt, comb=comb)
        else:
            Xt, Yt = self.step_no_fast(x=x, y=y, dt=dt, comb=comb)

        return Xt, Yt

    def step_euler(self, x, y, dt, dWt=None):
        if dWt is None:
            dWt = np.random.normal(size=(self.d, self.d)) * np.sqrt(dt)
        ind = np.eye(self.d)
        for i in range(self.n,self.d):
            ind[i,i] = 0
        sqrt_x = utils.cholesky(x)
        Xt = x + (self.alpha + self.b.dot(x) + x.dot(self.b.T))*dt + sqrt_x.dot(dWt).dot(ind) + ind.dot(dWt).dot(sqrt_x)
        Yt = y + utils.cholesky(Xt).dot(dWt).dot(self.rho.reshape(-1,1)).T[0]

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
        t, tmp_etb, tmp_int_etb = self.tmp_intgrl
        assert t == dt
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
        epsilon_sqr = self.epsilon * self.epsilon
        rho_q = self.rho[q]
        Xt = self.x_gen.step(x=x, q=q, dt=epsilon_sqr * dt)  # Generate Xt.
        dXt = Xt - x  # Calculate dXt.
        Yt = y + rho_q / self.epsilon * dXt[q]  # Calculate Yt.
        Yt[q] = y[q] + rho_q / (2 * self.epsilon) * (dXt[q, q] - epsilon_sqr * (self.d - 1) * dt)
        return Xt, Yt

    def step_L_bar_q(self, u, y, dt, dWt, q):
        rho_q = self.rho[q]
        # Update Y.
        Yt = y + rho_q * np.matmul(u.T, dWt[:, q])
        tmp = np.sum(dWt[:, q] * dWt[:, q] - dt)
        Yt[q] = Yt[q] + self.epsilon * rho_q / 2 * tmp
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
                dWt = np.random.normal(size=(2, self.d, self.d)) * np.sqrt(dt / 2)
            else:
                dWt = np.array(dWt)
                if dWt.shape != (2, self.d, self.d):  # If dWt is of shape (d, d).
                    assert dWt.shape == (self.d, self.d)
                    # Use the Brownian Bridge.
                    dWt_0 = dWt / 2 + np.random.normal(size=(self.d, self.d)) * np.sqrt(dt / 4)
                    dWt_1 = dWt - dWt_0
                    dWt = np.array([dWt_0, dWt_1])
            for q in range(1, self.n)[::-1]:
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt / 2, dWt=dWt[0], q=q)
            Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt, dWt=dWt[0]+dWt[1], q=0)
            for q in range(1, self.n):
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt / 2, dWt=dWt[1], q=q)
            return Ut, Yt

        elif comb == 'r' or comb == '2' or comb == 2:
            if zeta is None:
                zeta = np.random.rand(self.n - 1)
            else:
                assert len(zeta) == (self.n - 1)
            # Construct the combination order.
            seq_q = [0]
            for q in range(1, self.n):
                if zeta[q - 1] < .5:
                    seq_q.append(q)  # 96.6ns.
                else:
                    seq_q = [q] + seq_q  # 138ns.

            for q in seq_q:
                Ut, Yt = self.step_L_bar_q(u=Ut, y=Yt, dt=dt, dWt=dWt, q=q)
            return Ut, Yt
        else:
            pass
    
    def cal_tmp(self, t, alpha, b, num_int=200):
        tmp_etb = scipy.linalg.expm(t*b)
        tmp_int_etb = intgrl_etb(T=t, alpha=alpha, b=b, num_int=num_int)[-1]
        self.tmp_intgrl = [t, tmp_etb, tmp_int_etb]

    def character(self, Gamma, Lambda, Xt, Yt):
        return (np.exp(-1j*(np.trace(np.matmul(Gamma,Xt), axis1=1, axis2=2)+np.matmul(Yt,Lambda)))).mean()


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

