def test_cir_characteristic():
    import sys
    sys.path.append('./Processus-Wishart-513/')

    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm

    import cir

    k = 1 / 2
    a = 1 / 2
    sigma = 0.8
    T = 1
    x0 = 3 / 2
    num = 50000

    print(f'sigma^2 <= 4a is {sigma * sigma <= 4 * a}.')

    cir_gen = cir.CIR(k, a, sigma, x0=x0)
    lst_n = [20, 100, 200]

    xT_exact = cir_gen(T=T, n=1, num=num)[:, -1]  # The exact generated XT.
    char_exact = np.mean(np.exp(-1 * xT_exact))

    char_2_n = np.zeros(len(lst_n))
    char_3_n = np.zeros(len(lst_n))
    xT_2 = cir_gen(T=T, n=lst_n[0], num=num, method='2')[:, -1]
    xT_3 = cir_gen(T=T, n=lst_n[0], num=num, method='3')[:, -1]

    for i in range(len(lst_n)):
        xT_2 = cir_gen(T=T, n=lst_n[i], num=num, method='2')[:, -1]
        xT_3 = cir_gen(T=T, n=lst_n[i], num=num, method='3')[:, -1]
        char_2 = np.mean(np.exp(-1 * xT_2))
        char_3 = np.mean(np.exp(-1 * xT_3))

        char_2_n[i] = char_2
        char_3_n[i] = char_3

    plt.hist(xT_exact, density=True, bins=200, histtype='step', label='exact')
    plt.hist(xT_2, density=True, bins=200, histtype='step', label='2')
    plt.hist(xT_3, density=True, bins=200, histtype='step', label='3')
    plt.title('X of CIR')
    plt.legend()
    plt.show()

    plt.plot(char_exact, density=True, bins=200, histtype='step', label='exact')
    plt.plot(char_2, density=True, bins=200, histtype='step', label='2')
    plt.plot(char_3, density=True, bins=200, histtype='step', label='3')
    plt.title('characteristic of CIR')
    plt.legend()
    plt.show()


def test_wishart_processus():
    import numpy as np
    import matplotlib.pyplot as plt

    from wishart.simulation import Wishart

    x = np.array([[1, 1], [1, 2]])
    b = np.array([[1, 0], [0, 0]])
    a = np.array([[2, -1], [2, 1]])
    w = Wishart(x, 3, a, b)
    t = 1
    N = 30

    proc = w(t, N, x, b, a)
    interval = range(N + 1)

    plt.plot(interval, proc[0, :, 0, 0], label="[0,0]")
    plt.plot(interval, proc[0, :, 0, 1], label="[0,1]")
    plt.plot(interval, proc[0, :, 1, 1], label="[1,1]")
    plt.plot(interval, proc[0, :, 1, 0], label="[1,0]")
    plt.title('wishart')
    plt.legend()
    plt.show()

    proc2 = w.wishart_i(t, N, 1, x)
    interval = range(N + 1)

    plt.plot(interval, proc2[0, :, 0, 0], label="[0,0]")
    plt.plot(interval, proc2[0, :, 0, 1], label="[0,1]")
    plt.plot(interval, proc2[0, :, 1, 1], label="[1,1]")
    plt.plot(interval, proc2[0, :, 1, 0], label="[1,0]")
    plt.title('wishart i')
    plt.legend()
    plt.show()


def test_wishart_characteristic():
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    import sys
    sys.path.append('./Processus-Wishart-513/')

    import wishart

    def char_MC_N(gen, T, v, x=None, lst_N=[1], num=500, method='exact', **kwargs):
        lst_char = []
        for N in tqdm(lst_N):
            XT = gen(T=T, N=N, num=num, x=x, method=method, **kwargs) # of shape (num, d, d).
            tmp = np.matmul(v, XT)
            trace = np.trace(tmp, axis1=1, axis2=2)
    #         char = np.cumsum(np.exp(trace)) / np.arange(1, num+1)
            char = np.mean(np.exp(trace))
            lst_char.append(char)

        return np.array(lst_char)
    T = 10

    x = 0.4 * np.eye(3)
    a = np.eye(3)
    b = np.zeros((3, 3))
    alpha = 4.5

    num = 10000
    lst_N = np.array([1, 2, 4, 8, 10, 20])

    v = np.eye(3) * 0.05 * 1j

    lst_v = np.array([v])
    w_gen = wishart.Wishart(x, alpha, b=b, a=a)
    char_true = w_gen.character(T=T, v=lst_v, num_int=2000)[0]
    print(f'True value is {char_true}.')
    print('Calculating exact...')
    char_exact_N = char_MC_N(w_gen, T=T, v=v, lst_N=lst_N, num=num, method='exact', num_int=2000)
    print('Calculating 2nd order scheme...')
    char_2 = char_MC_N(w_gen, T=T, v=v, lst_N=lst_N, num=num, method='2', num_int=2000)
    print('Calculating 3rd order scheme...')
    char_3 = char_MC_N(w_gen, T=T, v=v, lst_N=lst_N, num=num, method='3', num_int=2000)
    print('Calculating euler scheme...')
    char_e = char_MC_N(w_gen, T=T, v=v, lst_N=lst_N, num=num, method='euler', num_int=2000)
    plt.axhline(y=np.real(char_true), color='r', label='True value')
    # plt.axhline(y=np.abs(char_exact), label='exact', alpha=.8, color='y')
    plt.plot(lst_N, np.real(char_exact_N), label='exact', alpha=.8)
    plt.plot(lst_N, np.real(char_2), label='2', alpha=.8)
    plt.plot(lst_N, np.real(char_3), label='3', alpha=.8)
    # plt.plot(lst_N, np.real(char_e), label='euler', alpha=.8)
    plt.legend()
    plt.xlabel('N')
    plt.title('Convergence of Wishart simulation methods')
#     plt.savefig('./wishart_cov.png')
    plt.show()


def test_cholesky():
    import numpy as np
    from wishart import utils

    a = np.array([[1,2,3],[2,5,6],[3,6,9]])
    c, k, p, r = utils.decompose_cholesky(a)
    print(c)
    print(k)
    print(p)
    print(r)
    b = np.zeros(a.shape)
    b[:r, :r] = c
    b[r:, :r] = k
    print(a)
    print(p.T.dot(b.dot(b.T)).dot(p))


def test_gs():
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
        S, X = model(num=num, N=10, T=T, ret_vol=True, method=method)
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
