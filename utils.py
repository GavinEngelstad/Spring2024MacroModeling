import jax.numpy as jnp
import numpy as np
import warnings
import jax


def newton(F, x0, tol=10e-16, maxiter=100, prnt=False):
    '''
    Uses newtons method to find roots. Relies on jax differentiation

    based on
    https://jax.quantecon.org/newtons_method.html
    '''
    x = x0
    J = jax.jacobian(F)
    err = tol + 1
    n = 0
    @jax.jit
    def q(x):
        return x - jnp.linalg.solve(J(x), F(x)) # update rule

    while err > tol and n < maxiter:
        n += 1
        y = q(x)
        err = np.linalg.norm(x - y)
        x = y
        if prnt: print(f'Iteration {n}: Error: {err}')
    return x


def simp_newton(F, x0, tol=10e-16, maxiter=100, prnt=False):
    '''
    Uses newtons method to find roots of a R1 -> R1 function
    '''
    x = x0
    dF = jax.grad(F)
    err = tol + 1
    n = 0
    @jax.jit
    def q(x):
        return x - F(x)/dF(x) # update rule

    while err > tol and n < maxiter:
        n += 1
        y = q(x)
        err = np.abs(x-y)
        x = y
        if prnt: print(f'Iteration {n}: Error: {err}')
    return x


def zero_if_close(A, tol=1e-7):
    '''
    Replace values in an array that are close to 0 with 0
    '''
    A[np.abs(A) < tol] = 0
    return A


def solve_system(A, B, C, E, P0=None, MAXIT = 1000, tol=1e-7):
    '''
    Solve the system using linear time iteration like in Rendahl (2017).

    A: Matrix corresponding to time t+1
    B: Matrix corresponding to time t
    C: Matrix corresponding to time t-1
    E: Matrix corresponding to error terms
    P0: inital guess at the system solution

    Code (mostly) from Alisdair McKay
    '''
    # Solve the system using linear time iteration as in Rendahl (2017)
    if P0 is None:
        P = np.zeros(A.shape)
    else:
        P = P0

    S = np.zeros(A.shape)
    err = tol+1
    it = 0

    while err > tol:
        if it >= MAXIT:
            break
        it += 1

        P = -np.linalg.lstsq(B + A@P, C, rcond=None)[0]
        S = -np.linalg.lstsq(B + C@S, A, rcond=None)[0]
        err = np.max(np.abs(C + B@P + A@P@P))

    # test Blanchard-Kahn conditions
    if np.max(np.linalg.eig(P)[0]) > 1:
        raise RuntimeError("Model does not satisfy BK conditions -- non-existence")

    if np.max(np.linalg.eig(S)[0]) > 1:
        raise RuntimeError("Model does not satisfy BK conditions -- mulitple stable solutions")
    
    if it >= MAXIT:
        warnings.warn(f'LTI did not converge. Error: {err}')

    P = zero_if_close(P, tol)

    # Impact matrix
    #  Solution is x_{t}=P*x_{t-1}+Q*eps_t
    Q = -np.linalg.inv(B + A@P) @ E

    return P, Q


def impulse_response(P, Q, T, i, sigma):
    '''
    Run an impulse response

    P: Policy function P (States)
    Q: Policy function Q (Random Variables)
    T: Length of impulse response
    i: index of affected variable
    sigma: size of shock
    '''
    # Calculate an impulse response
    irf = np.zeros((P.shape[0], T))
    irf[:,0] = sigma*Q[:, i] # change epsilon

    for t in range(1,T):
        irf[:,t] = P @ irf[:,t-1] # update from last period

    return irf


def simulate(P, Q, T, sigma, seed=None):
    '''
    Simulate the economy over a period of time

    P: Policy function P (States)
    Q: Policy function Q (Random Variables)
    T: Length of simulation
    sigma: standard deviation of shock errors
    '''
    # set seed
    if seed != None:
        np.random.seed(seed)

    # initialize array
    sim = np.zeros((P.shape[0], T))
    sim[:,0] = Q @ (sigma*np.random.normal(size=sigma.size))

    # update rule
    for t in range(1, T):
        sim[:,t] = P @ sim[:,t-1] + Q @ (sigma*np.random.normal(size=sigma.size)) # update from last period

    return sim
