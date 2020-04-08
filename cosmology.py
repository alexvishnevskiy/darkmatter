import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import integrate
import math
import opt

t = np.loadtxt('jla_mub.txt', dtype=np.float)
z = t[:, 0]
nu = t[:, 1]
c = 3e11


def fun(x, omega):
    return 1 / np.sqrt((1 - omega) * (1 + x) ** 3 + omega)


def fun1(x, omega):
    return -(5/(2*2.30))*(((x+1)**3)-1)/((omega - (omega-1)*((x+1)**3))**(3/2))


def jacobi(x, omega, H):
    j = []
    for i in x:
        k = integrate.quad(fun, 0, i, args=omega)[0]
        k1 = integrate.quad(fun1, 0, i, args=omega)[0]
        j.append(k1 / k)
    f1 = (5/(2.30*H)) * np.ones(31)
    return np.vstack((j, f1))


def function(x, omega, H):
    f = np.array([integrate.quad(fun, 0, i, args=omega)[0] for i in x])
    return 5*np.log10((c/H)*(1+np.array(x))*f) - 5


r = opt.gauss_newton(
    y = nu,
    f = lambda *x: function(z, *x),
    j = lambda *x: jacobi(z, *x),
    x0 = np.array([0.5, 50])
)
print(r)


plt.figure(figsize = [10, 9])
plt.title(r'$\mu$-z')
plt.plot(z, nu, 'x')
plt.plot(z, function(z, 0.73608177, 70.4483552), label = 'Gauss-Newton')
plt.grid(True)
plt.xlabel(r' $ \mu $', labelpad = 10, fontsize = 15)
plt.ylabel(r' z', labelpad = 10, fontsize = 15)
plt.legend()
plt.savefig('mu-z.png')
plt.figure(figsize = [10, 9])
plt.xlabel('Шаг', size=8)
plt.ylabel('Сумма потерь', size=8)
plt.plot(np.arange(1,r.nfev+1), r.cost)
plt.grid()
plt.title('Функция потерь', size=10)
plt.savefig('cost.png')

data = {
   "Gauss-Newton": {"H0": r.x[0], "Omega": r.x[1], "nfev": r.nfev},
}



with open("parameters.json", "w") as f:
    json.dump(data, f)
