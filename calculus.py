"""
calculus.py
"""

import scipy.optimize as opt

def dummy():
    """ dummy functions for template file
    just the same function that Dr Thomay made"""
    return 0

def secant_wrapper(func, x0, x1, args=(), tol=1e-6, maxiter=50):
    """
    Wrapper for the secant method using scipy.optimize.root_scalar.

    Parameters:
    func : The function for which the root is to be found.
    x0 : The first initial guess (Ideally close to, but less than, the root)
    x1 : The second initial guess (Ideally close to, but greater than, the root)
    args : Additional arguments to pass to the function. Must be a tuple
    tol : Optional tolerance deciding when to stop function (default is 1e-6).
    maxiter : The maximum number of iterations if convergence is not met (default is 50).

    Returns:
    A dictionary containing:
        - 'root': The estimated root.
        - 'converged': Boolean indicating whether the method converged.
        - 'iterations': Number of iterations performed.
        - 'function_calls': Number of function evaluations performed.
    """

    #Use the secant method
    res = opt.root_scalar(
        func,
        args=args,
        method="secant",
        x0=x0,
        x1=x1,
        xtol=tol,
        maxiter=maxiter)

    #Return a callable dictionary
    return {
        "root": res.root if res.converged else None,
        "converged": res.converged,
        "iterations": res.iterations,
        "function_calls": res.function_calls}
