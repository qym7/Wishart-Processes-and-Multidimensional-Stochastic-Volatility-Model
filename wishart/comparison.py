import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from wishart import utils
from wishart.simulation import Wishart

def exact_exp_trace(x, alpha, b, a, t, vr, vi, num_int=200):
    if x is None:
        x = x
    d = x.shape[0]
    qt = utils.integrate(t, b, a, d=d, num_int=num_int)
    mt = scipy.linalg.expm(t * b)
    v = vr + 1j * vi
    exp = np.exp(np.trace(v.dot(np.linalg.inv(np.eye(d) - 2 * qt.dot(v))).dot(mt).dot(x).dot(mt.T)))

    return exp / np.power(np.linalg.det(np.eye(d) - 2 * qt.dot(v)), alpha / 2)

def compare_exp_trace(x, alpha, b, a, t, vr, vi, n_simu=100, num_int=200, method="exact"):
    wishart = Wishart(x, alpha, a=a, b=b)
    simu_res = wishart(t, b, a, N=1, num=n_simu, x=x, method=method)
    simu_res = np.exp(np.trace(simu_res.dot(vr+1j*vi))).mean(axis=0)
    exact_res = exact_exp_trace(x, alpha, b, a, t, vr, vi, num_int=num_int)

    return exact_res - simu_res

def generate_random_sp(d):
    A = np.random.randn(d, d)
    return np.dot(A, A.transpose())

def show_diff(l, x, alpha, b, a, t, n_simu=100, num_int=200, n_test=10, method="exact"):
    d = x.shape[0]
    vr = np.zeros((d,d))
    # vi = 1j * np.ones((d,d))
    # vi = 1j * np.eye(d)
    for k in range(n_test):
        vi = 1j * generate_random_sp(d)
        diff = [compare_exp_trace(x, alpha, b, a, t, vr, vi*i, n_simu=n_simu, num_int=num_int, method=method) for i in l]
        plt.plot(l, diff)

    plt.show()

def compare_diff(l, x, alpha, b, a, t, n_simu=100, num_int=200):
    d = x.shape[0]
    vr = np.zeros((d,d))*1000
    for k in ["exact", "2", "3"]:
        vi = 1j * generate_random_sp(d)
        diff = [compare_exp_trace(x, alpha, b, a, t, vr, vi*i, n_simu=n_simu, num_int=num_int, method=k) for i in l]
        print(diff)
        plt.plot(l, diff, label=k)
        plt.legend()
    plt.show()


if __name__ == '__main__':
    t = 2.0
    x = np.array([[1,0],[0,2]])
    b = np.array([[1,1],[1,1]])
    a = np.array([[2,-1],[2,1]])
    alpha = 3
    l = range(1, 20)
    # show_diff(l, x, alpha, b, a, t, n_simu=50, num_int=200, method="3")
    compare_diff(l, x, alpha, b, a, t, n_simu=50, num_int=200)