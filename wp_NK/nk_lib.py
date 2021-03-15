import numpy as np
import pennylane as qml
from collections.abc import Iterable
import matplotlib.pyplot as plt
import seaborn as sns

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


def add_noise_channel(operation_list, noise_channel, idling_gate_noise_channel=None, adjoint=False):
    """This is not tested yet! A less hacky version to add noise to the circuit, copied from qml.inv 
    Args:
      operation_list (list<qml.Operation>):
      noise_channel (callable): Signature: (gate_parameters, wires) -> list<qml.Operation>
        Warning: The callable needs to handle gate_parameters of any length, incl. 0!
      idling_gate_noise_channel (callable): Signature: (wires) -> list<qml.Operation>
    """
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
    elif isinstance(operation_list, qml.tape.QuantumTape):
        return add_noise_channel(operation_list.operations, noise_channel, idling_gate_noise_channel, adjoint)

    elif not isinstance(operation_list, Iterable):
        raise ValueError(f"The provided operation_list is not iterable: {operation_list}")

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
            active_wires = set()
            time_step = 0
            if adjoint:
                operation_list = operation_list[::-1]
            for o in operation_list:
                if set(o.wires).intersection(active_wires):
                    if idling_gate_noise_channel is not None:
                        inactive_wires = list(set(device.wires).difference(active_wires))
                        (c.queue() for c  in idling_gate_noise_channel(inactive_wires))
                    time_step += 1
                    active_wires = set()
                o.queue()
                if (not adjoint and o.inverse) or (adjoint and not o.inverse):
                    o.inv()
                # TODO: add a case distinction such that noise is not added to noise
                (c.queue() for c in noise_channel(o.data, o.wires))

        return tape

    for op in operation_list:
        qml.QueuingContext.remove(op)

    all_ops = []
    active_wires = set()
    time_step = 0
    if adjoint:
        operation_list = operation_list[::-1]
    for op in operation_list:
        if set(op.wires).intersection(active_wires):
            if idling_gate_noise_channel is not None:
                inactive_wires = list(set(device.wires).difference(active_wires))
                channels = idling_gate_noise_channel(inactive_wires)
                (qml.QueuingContext.append(c) for c in channels)
                (all_ops.append(c) for c in channels)
            active_wires = set()
            time_step += 1

        channels = noise_channel(op.data, op.wires)
        if (not adjoint and op.inverse) or (adjoint and not op.inverse):
            op.inv()
        qml.QueuingContext.append(op)
        (qml.QueuingContext.append(c) for c in channels)
        all_ops.append(op)
        (all_ops.append(c) for c in channels)

    return all_ops

def mitigate_global_depolarization(kernel_matrix, num_wires, strategy='average', use_entries=None, return_noise_estimates=False):
    """Estimate the noise rate of a global depolarizing noise model based on the diagonal entries of a kernel
    matrix and mitigate the effect of said noise model.
    Args:
      kernel_matrix (ndarray): Noisy kernel matrix.
      num_wires (int): Number of wires/qubits that was used to compute the kernel matrix.
      strategy='average' ('average'|'split_channel'|None): Details of the noise model and strategy for mitigation.
        'average': Compute the noise rate based on the diagonal entries in use_entries, average if applicable.
        'split_channel': Assume a distinct effective noise rate for the embedding circuit of each feature vector.
        None: Don't do anything.
      use_entries=None (list<int>): Indices of diagonal entries to use if strategy=='average'. Set to all if None.
      return_noise_estimates=False (bool): Whether or not to return the estimated noise rate(s).
    Returns:
      mitigated_matrix (ndarray): Mitigated kernel matrix.
      noise_rates (ndarray): Determined noise rates, meaning depends on kwarg strategy, only returned if activated.
    Comments:
      If strategy is 'average', the diagonal entries with indices use_entries have to be measured on the QC.
      If it is 'split_channel', all diagonal entries are required.
    """
    dim = 2**num_wires

    if strategy is None:
        return kernel_matrix, None

    elif strategy=='average':
        if use_entries is None:
            diagonal_elements = np.diag(kernel_matrix)
        else:
            diagonal_elements = np.diag(kernel_matrix)[use_entries]
        noise_rates = (1 - diagonal_elements) * dim / (dim - 1)
        eff_noise_rate = np.mean(noise_rates)
        mitigated_matrix = (kernel_matrix - eff_noise_rate / dim) / (1 - eff_noise_rate)

    elif strategy=='split_channel':
        noise_rates = np.clip((1 - np.diag(kernel_matrix)) * dim / (dim - 1), 0., 1.)
#         print(noise_rates)
        noise_rates = 1-np.sqrt(1-noise_rates)
#         print(noise_rates)
        n = len(kernel_matrix)
        inverse_noise = -np.outer(noise_rates, noise_rates)\
            + noise_rates.reshape((1, n))\
            + noise_rates.reshape((n, 1))
#         print(inverse_noise)
        mitigated_matrix = (kernel_matrix - inverse_noise / dim) / (1 - inverse_noise)
    if return_noise_estimates:
        return mitigated_matrix, noise_rates
    else:
        return mitigated_matrix


class noisy_kernel(qml.kernels.EmbeddingKernel):
    """adds noise to a QEK
    Args:
      ansatz (callable): See qml.kernels.EmbeddingKernel
      device (qml.Device, Sequence[Device]): See qml.kernels.EmbeddingKernel
      noise_channel (callable): Noise channel to be applied on the level of noise_application_level.
        Signature: (*args_noise_channel, operation_data, wires) -> list<qml.Operation>
      args_noise_channel (dict): arguments to be passed to the noise_channel.
      noise_application_level (str):
        'global': Apply noise after the full circuit.
        'per_embedding': Apply noise after the embedding and the inverted embedding each.
        'per_gate': Apply noise after each gate (incompatible with noise_channels acting on all qubits).
      idling_gate_noise_channel=None (callable|bool): Noise channel to be applied after an idling gate.
        Signature: (wires) -> list<qml.Operation>
        If True, use noise_channel with default data kwarg, if False, do not apply noise to idle qubits.
    Comments:
      WARNING: The noise_channel must be able to take any number of operation_data arguments that
               occurs in the ansatz, e.g. 0 for unparametrized gates, 1 for simple rotations and so on.
               When using noise_application_level in ('global', 'per_embedding') or 
               idling_gate_noise_channel==True, a default must be set for the operation_data argument(s).
    """

    def __init__(
        self,
        ansatz,
        device,
        noise_channel,
        args_noise_channel,
        noise_application_level,
        idling_gate_noise_channel=False,
        **kwargs
    ):
        self.probs_qnode = None
        self.noise_channel = noise_channel
        self.args_noise_channel = args_noise_channel
        self.noise_application_level = noise_application_level
        # Make the noise_channel callable on operator data and wires but incorporate args_noise_channel
        self.noise_channel_mapped = lambda data, wires: noise_channel(*args_noise_channel, data, wires=wires)
        if idling_gate_noise_channel is True:
            self.idling_gate_noise_channel = lambda wires: self.noise_channel(*args_noise_channel, wires=wires)
        if idling_gate_noise_channel is False:
            self.idling_gate_noise_channel = None
        else:
            self.idling_gate_noise_channel = idling_gate_noise_channel

        def circuit(x1, x2, params, **kwargs):
            if self.noise_application_level == 'per_gate':
                add_noise_channel(
                    ansatz(x1, params, **kwargs),
                    self.noise_channel_mapped,
                    self.idling_gate_noise_channel,
                )
            else:
                ansatz(x1, params, **kwargs)

            if self.noise_application_level == 'per_embedding':
                # Invoke noise channel with default kwarg data.
                self.noise_channel(*self.args_noise_channel, wires=device.wires)

            elif self.noise_application_level == 'per_gate':
                add_noise_channel(
                    ansatz(x2, params, **kwargs),
                    self.noise_channel_mapped,
                    self.idling_gate_noise_channel,
                    adjoint=True,
                )
            else:
                qml.inv(ansatz(x2, params, **kwargs))

            if self.noise_application_level in ('per_embedding', 'global'):
                # Invoke noise channel with default kwarg data.
                self.noise_channel(*self.args_noise_channel, wires=device.wires)

            return qml.probs(wires=device.wires)

        self.probs_qnode = qml.QNode(circuit, device, **kwargs)

def visualize_kernel_matrices(kernel_matrices, draw_last_cbar=False):
    num_mat = len(kernel_matrices)
    width_ratios = [1]*num_mat+[0.2]*int(draw_last_cbar)
    figsize = (num_mat*5+draw_last_cbar, 5)
    num_axes = num_mat+draw_last_cbar
    fig,ax = plt.subplots(1, num_axes, figsize=figsize, gridspec_kw={'width_ratios': width_ratios})

    for i, kernel_matrix in enumerate(kernel_matrices):
        plot = sns.heatmap(
            kernel_matrix,
            vmin=0,
            vmax=1,
            xticklabels='',
            yticklabels='',
            ax=ax[i],
            cmap='Spectral',
            cbar=False
        )
    if draw_last_cbar:
        ch = plot.get_children()
        fig.colorbar(ch[0], ax=ax[-2], cax=ax[-1])
