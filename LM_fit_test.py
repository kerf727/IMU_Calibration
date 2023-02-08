import numpy as np
import scipy.optimize as opt

# Initial guess for parameters to optimize
x_init = [0, 0, 0]

# Model to fit to
def y(x):
    return (x[0]**2 + x[1]**2 + x[2]**2)**0.5

# Real data to fit model to
ys = [1.3,
      1.1,
      1.2,
      1.0,
      0.9]

def fun(x):
    return y(x) - ys

result = opt.least_squares(fun, x_init, method='lm')#, ftol=1e-9, xtol=1e-9)
print("Success: {}".format(result.success))
print("{}".format(result.message))
print("x0 = %.3f" % (result.x[0]))
print("x1 = %.3f" % (result.x[1]))
print("x2 = %.3f" % (result.x[2]))