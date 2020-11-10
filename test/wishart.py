import numpy as np
import matplotlib.pyplot as plt

from wishart.wishart import Wishart

x = np.array([[1,1],[1,2]])
b = np.array([[1,0],[0,0]])
a = np.array([[2,-1],[2,1]])
w = Wishart(x, 3, a, b)
t = 1
N=30

proc = w(t, N, x, b, a)
interval = range(N+1)

plt.plot(interval, proc[0, :, 0, 0], label="[0,0]")
plt.plot(interval, proc[0, :, 0, 1], label="[0,1]")
plt.plot(interval, proc[0, :, 1, 1], label="[1,1]")
plt.plot(interval, proc[0, :, 1, 0], label="[1,0]")
plt.legend()
plt.show()

proc2 = w.wishart_i(t, N, 1, x)
interval = range(N+1)

plt.plot(interval, proc2[0, :, 0, 0], label="[0,0]")
plt.plot(interval, proc2[0, :, 0, 1], label="[0,1]")
plt.plot(interval, proc2[0, :, 1, 1], label="[1,1]")
plt.plot(interval, proc2[0, :, 1, 0], label="[1,0]")
plt.legend()
plt.show()