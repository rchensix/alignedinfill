# Ruiqi Chen
# March 10, 2020

import numpy as np
from typing import Callable, Tuple

# Tries every value in domain to find minimum. Returns (min, argmin)
def brute_force_1d(func: Callable[[np.ndarray], np.ndarray], domain: np.ndarray) -> Tuple[float, float]:
    val = func(domain)
    argind = np.argmin(val)
    ind = np.unravel_index(argind, val.shape)
    return val[ind], domain[ind]

def optimizer_test():
    def f(x):
        return x**3
    val, arg = brute_force_1d(f, np.linspace(-2, 2, 10))
    tol = 1e-12
    assert np.abs(val + 8) < tol, 'minimum is at -8, not ' + val
    assert np.abs(arg + 2) < tol, 'argmin is at -2, not ' + arg
    print('passed optimizer_test')

def main():
    optimizer_test()

if __name__ == '__main__':
    main()