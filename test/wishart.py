import numpy as np

from wishart.wishart import Wishart

x = np.array([[1,0],[1,2]])
b = np.array([[3,3],[3,3]])
a = np.array([[2,-1],[2,1]])
w = Wishart(x, 3, a, b)
t = 1
print(w.wishart_e(t, x))
print(w.wishart_i(t, x, 1))
print(w(x,t,b,a))