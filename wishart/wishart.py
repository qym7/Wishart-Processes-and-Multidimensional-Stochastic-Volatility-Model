import numpy as np

from wishart import linalg

class Wishart():
    def __init__(self, x, alpha, d=None, b=0, a=None):
        '''
        :param x:
        :param d:
        :param alpha:
        :param b:
        :param a:
        '''
        self.x = x
        self.alpha = alpha
        self.d = d or x.shape[0]
        self.b = b
        self.a = a or np.eye(d)

    def wishart_e(self):
        pass

    def wishart_i(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
