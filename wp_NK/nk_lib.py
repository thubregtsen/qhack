import numpy as np

def closest_psd_matrix(K, solver=None, fix_diagonal=False):
    """Return the closest positive semidefinite matrix to the given kernel matrix.

    Args:
        K (array[float]): Kernel matrix assumed to be symmetric
        solver (str, optional): Solver to be used by cvxpy. Defaults to CVXOPT.
        fix_diagonal (bool): Whether to fix the diagonal of the matrix to ones.

    Returns:
        array[float]: closest positive semidefinite matrix in Frobenius norm.
    """
    try:
        import cvxpy as cp
        if solver is None:
            solver = cp.CVXOPT
    except ImportError:
        print("CVXPY is required for this post-processing method.")
        return K

    wmin = np.min(np.linalg.eigvals(K))
    if wmin >= 0:
        return K

    X = cp.Variable(K.shape, PSD=True)
    objective_fn = cp.norm(X - K, "fro")
    if fix_diagonal:
        constraint = [cp.diag(X) == 1.]
    else:
        constraint = []
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)

    try:
        problem.solve(solver=solver, feastol=1e-6)
    except Exception as e:
        problem.solve(verbose=True, solver=solver)

    return X.value
