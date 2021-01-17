import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg
import importlib
from tqdm import tqdm

import sys
sys.path.append('..')

import sampling
import cir
import wishart
# from application import GS_model
from application import GS_model


def price_mc(model, num, r, T, K, N, method):
    S, X = model(num=num, N=10, T=T, ret_X=True, method=method)
    ST = S[:, -1]
    ST_M = np.max(ST, axis=1)
    prix = (K-ST_M).clip(0) * np.exp(-r*T)
    prix = prix.mean()
    return prix

S0 = np.array([100, 100])
r = .02
X0 = np.array([[.04, .02], [.02, .04]])
alpha = 4.5
a = np.eye(2) * 0.2
b = np.eye(2) * 0.5
T = 1
K = 120
num = 50000
# lst_N = np.array([1, 2, 4, 8, 10, 20, 25])
lst_N = np.array([1, 2, 4, 8, 10, 20])
# lst_N = np.array([1, 10, 100])

model = GS_model(S0, r, X0, alpha, a=a, b=b)
lst_prix_exact = np.zeros_like(lst_N, dtype=float)
lst_prix_2 = np.zeros_like(lst_N, dtype=float) 
lst_prix_3 = np.zeros_like(lst_N, dtype=float)
lst_prix_e = np.zeros_like(lst_N, dtype=float)

it_lst = tqdm(range(len(lst_N)))
for i in it_lst:
    N = lst_N[i]
    it_lst.set_postfix({'calculating': 'exact...'})
    prix = price_mc(model, num=num, T=T, K=K, N=N, r=r, method='exact')
    lst_prix_exact[i] = prix
    it_lst.set_postfix({'calculating': 'scheme 2...'})
    prix = price_mc(model, num=num, T=T, K=K, N=N, r=r, method='2')
    lst_prix_2[i] = prix
    it_lst.set_postfix({'calculating': 'scheme 3...'})
    prix = price_mc(model, num=num, T=T, K=K, N=N, r=r, method='3')
    lst_prix_3[i] = prix
    it_lst.set_postfix({'calculating': 'scheme euler...'})
    prix = price_mc(model, num=num, T=T, K=K, N=N, r=r, method='euler')
    lst_prix_e[i] = prix
    

    
plt.plot(lst_N, lst_prix_exact, label='exact')
plt.plot(lst_N, lst_prix_2, label='2')
plt.plot(lst_N, lst_prix_3, label='3')
# plt.plot(lst_N, lst_prix_e, label='euler')
plt.legend()
plt.show()