#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    aux = circuit(weights)
    shift_1 = np.zeros([5], dtype = np.float64)
    shift_2 = np.zeros([5], dtype = np.float64)
    for i in range(5):
        shift_1[i] = .5*np.pi
        aux_f = circuit(weights + shift_1)
        aux_b = circuit(weights - shift_1)
        gradient[i] = aux_f - aux_b
        gradient[i] *= .5

        hessian[i][i] = aux_f + aux_b - 2*aux
        hessian[i][i] *= .5
        
        for j in range(i):
            shift_2[j] = .5*np.pi
            hessian[i][j] = (circuit(weights + shift_1 + shift_2)
                             + circuit(weights - shift_1 - shift_2)
                             - circuit(weights + shift_1 - shift_2)
                             - circuit(weights - shift_1 + shift_2))
            hessian[i][j] *= .25
            hessian[j][i] = hessian[i][j]
            shift_2[j] = 0.0
        shift_1[i] = 0.0
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
