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
    pass


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
    pass
