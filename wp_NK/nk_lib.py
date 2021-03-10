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


def add_noise_channel(operation_list, noise_channel):
    """This is not tested yet! A less hacky version to add noise to the circuit, copied from qml.inv """
    if isinstance(operation_list, qml.operation.Operation):
        operation_list = [operation_list]
    elif operation_list is None:
        raise ValueError(
            "None was passed as an argument to add_noise_channel. "
            "This could happen if adding noise to a template without the template decorator is attempted."
        )
    elif callable(operation_list):
        raise ValueError(
            "A function was passed as an argument to add_noise_channel. "
            "This could happen if adding noise to a template function is attempted. "
            "Please use add_noise_channel on the function including its arguments, "
            "as in add_noise_channel(template(args), channel)."
        )
#     elif isinstance(operation_list, qml.tape.QuantumTape):
#         operation_list.inv()
#         return operation_list
    elif not isinstance(operation_list, Iterable):
        raise ValueError("The provided operation_list is not iterable.")

    non_ops = [
        (idx, op)
        for idx, op in enumerate(operation_list)
        if not isinstance(op, qml.operation.Operation)
    ]

    if non_ops:
        string_reps = [" operation_list[{}] = {}".format(idx, op) for idx, op in non_ops]
        raise ValueError(
            "The given operation_list does not only contain Operations."
            + "The following elements of the iterable were not Operations:"
            + ",".join(string_reps)
        )

    if qml.tape_mode_active():
        for op in operation_list:
            try:
                # remove the queued operation to add noise to
                # from the existing queuing context
                qml.tape.QueuingContext.remove(op)
            except KeyError:
                # operation to be inverted does not
                # exist on the queuing context
                pass

        with qml.tape.QuantumTape() as tape:
            for o in operation_list:
                o.queue()
                if o.inverse:
                    o.inv()
                (c.queue() for c in noise_channel(o.wires))

        return tape

    for op in operation_list:
        qml.QueuingContext.remove(op)

    all_ops = []
    for op in operation_list:
        qml.QueuingContext.append(op)
        (qml.QueuingContext.append(c) for c in noise_channel(op.wires))
        all_ops.append(op)
        (all_ops.append(c) for c in noise_channel(op.wires))

    return all_ops
