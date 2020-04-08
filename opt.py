import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import integrate
import math
from collections import namedtuple

Result = namedtuple('Result', ('nfev', 'cost', 'x'))
Result.__doc__ = """Результаты оптимизации
Attributes
----------
nfev : int
    Полное число вызовов можельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива меньше nfev
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


H_0 = 50
omega = 0.5
c = 3e11

t = np.loadtxt('jla_mub.txt', dtype=np.float)
z = t[:, 0]
nu = t[:, 1]


def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    e = j(*x0)
    r = y - f(*x0)
    x = x0 - k * np.linalg.inv(e @ e.T) @ e @ r
    cost = []
    cost.append(np.sum(0.5 * (y - f(*x0)) ** 2))
    nfev = 1
    while np.linalg.norm(x-x0)/(np.linalg.norm(x0)) > tol:
        x0 = x
        e = j(*x0)
        r = y - f(*x0)
        x = x0 - k * np.linalg.inv(e@e.T)@e@ r
        cost.append(np.sum(0.5 * (y-f(*x0))**2))
        nfev += 1
    return Result(nfev, cost, x)

def lm(y,f,j, x0, lmbd0=1e-2, mu=2, tol=1e-4):
    x = x0
    x1 = x0
    cost = []
    nfev = 0
    while True:
        nfev += 1
        x0 = x
        x0_1 = x1
        e = j(*x0)
        e1 = j(*x0_1)
        r = y - f(*x0)
        r1 = y - f(*x0_1)
        delta_x = np.linalg.inv(e @ e.T + lmbd0 * np.eye(2)) @ e @ r
        delta_x1 = np.linalg.inv(e1 @ e1.T + lmbd0 / mu * np.eye(2)) @ e1 @ r1
        x = x0 + delta_x
        x1 = x0_1 + delta_x1
        F0 = np.sum(0.5 * (y - f(*x0)) ** 2)
        F = np.sum(0.5 * (y - f(*x)) ** 2)
        F1 = np.sum(0.5 * (y - f(*x1)) ** 2)
        cost.append(np.sum(0.5 * (y - f(*x0)) ** 2))
        if F1 <= F0:
            lmbd0 = lmbd0/mu
        elif F1 > F0 and F1 <= F:
            lmbd0 = lmbd0
        else:
            while F1<= F0:
                lmbd0 = lmbd0*mu
        delta_x = np.linalg.inv(e @ e.T + lmbd0 * np.eye(2)) @ e @ r
        delta_x1 = np.linalg.inv(e1 @ e1.T + lmbd0 / mu * np.eye(2)) @ e1 @ r1
        if np.linalg.norm(x-x0)/(np.linalg.norm(x0)) <= tol:
            break
        x += delta_x
        x1 += delta_x1
    cost = np.array(cost)
    return Result(nfev, cost, x)

