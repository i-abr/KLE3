from jax import jacfwd, jit
from jax.ops import index, index_add, index_update
import jax.numpy as np
from jax.lax import cond
import scipy

Hz = 10.
@jit
def f(x, u_free):
    u = u_free
    x1, x2 = x
    xdot = u[0]
    ydot = u[1] 
    xdot = np.array([xdot, ydot])
    xnew = x + xdot/Hz
    return xnew

