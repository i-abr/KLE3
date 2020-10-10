from jax import jacfwd, jit
from jax.ops import index, index_add, index_update
import jax.numpy as np
from jax.lax import cond
import scipy

Hz = 10.
@jit
def f(x, u_free):
    u = np.tanh(u_free)
    x1, x2, th, x1dot, x2dot, thdot = x
    xddot = -np.sum(u) * np.sin(th)
    yddot = np.sum(u) * np.cos(th) - 1.
    thddot = 2.0*(u[0] - u[1])
    xdot = np.array([x1dot, x2dot, thdot, xddot, yddot, thddot])
    xnew = x + xdot/Hz
    return xnew

def wrap2pi(th):
    x = np.fmod(th + np.pi, 2.0*np.pi)
    x = cond(x < 0, x, lambda x: x+2.0*np.pi, x, lambda x: x)
    return x - np.pi

# LQR controller
x_eq, u_eq = np.zeros(6), np.ones(2)*0.5
lqr_config = {
    'A' : jacfwd(f)(x_eq, u_eq),
    'B' : jacfwd(f, argnums=1)(x_eq, u_eq),
    'Q' : np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    'R' : np.diag(0.001*np.ones(2))
}

def get_lqr_from_config(config):
    A, B = config['A'], config['B']
    Q, R = config['Q'], config['R']
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    k_lqr = np.linalg.inv(B.T.dot(P).dot(B) + R).dot(B.T.dot(P).dot(A))
    return k_lqr

# equilibrium stability policy 
k_lqr = get_lqr_from_config(lqr_config)
@jit
def pi(x):
    xmod = index_update(x, 2, wrap2pi(x[2]))
    return -np.dot(k_lqr, xmod)
