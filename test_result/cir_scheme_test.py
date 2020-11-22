import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg
import importlib
from tqdm.notebook import tqdm

import sys
sys.path.append('./Processus-Wishart-513/')

import sampling
import cir
import wishart



k = 1/2
a = 1/2
sigma = 0.8
T = 1
x0 =  3/2
num = 50000

print(f'sigma^2 <= 4a is {sigma*sigma <= 4*a}.')

cir_gen = cir.CIR(k, a, sigma, x0=x0)

# lst_n = [20, 200, 500, 1000, 1500, 2000]
# lst_n = np.arange(1, 11)
lst_n = [20, 100, 200]

xT_exact = cir_gen(T=T, n=1, num=num)[:, -1] # The exact generated XT.
char_exact = np.mean(np.exp(-1*xT_exact))

char_2_n = np.zeros(len(lst_n))
char_3_n = np.zeros(len(lst_n))
xT_2 = cir_gen(T=T, n=lst_n[0], num=num, method='2')[:, -1]
xT_3 = cir_gen(T=T, n=lst_n[0], num=num, method='3')[:, -1]
for i in range(len(lst_n)):
    xT_2 = cir_gen(T=T, n=lst_n[i], num=num, method='2')[:, -1]
    xT_3 = cir_gen(T=T, n=lst_n[i], num=num, method='3')[:, -1]
    char_2 = np.mean(np.exp(-1*xT_2))
    char_3 = np.mean(np.exp(-1*xT_3))
    
    char_2_n[i] = char_2
    char_3_n[i] = char_3
    
# plt.axhline(y=char_exact, color='r', label='exact')
# plt.plot(lst_n, char_2_n, label='2')
# plt.plot(lst_n, char_3_n, label='3')
# plt.legend()
# plt.show()


plt.hist(xT_exact, density=True, bins=200, histtype='step', label='exact')
plt.hist(xT_2, density=True, bins=200, histtype='step', label='2')
plt.hist(xT_3, density=True, bins=200, histtype='step', label='3')
plt.legend()
plt.show()