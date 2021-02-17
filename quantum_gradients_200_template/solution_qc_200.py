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

    def gradient_and_diagonal_hessian_elements(circuit, weights, index, f_theta):
        shift = np.pi/2
        shifted_weights = weights.copy()
        shifted_weights[index] += shift
        f_plus = circuit(shifted_weights)
        shifted_weights = weights.copy()
        shifted_weights[index] -= shift
        f_minus = circuit(shifted_weights)
        gradient = (f_plus - f_minus)/2
        hessian_element = (f_plus - 2 * f_theta + f_minus)/2
        return (gradient, hessian_element)

    def non_diagonal_hessian_elements(circuit, weights, index_1, index_2):
        shifted_weights = weights.copy()
        shifted_weights[index_1] += np.pi/2
        shifted_weights[index_2] += np.pi/2
        f_0 = circuit(shifted_weights)

        shifted_weights = weights.copy()
        shifted_weights[index_1] += np.pi/2
        shifted_weights[index_2] -= np.pi/2
        f_1 = circuit(shifted_weights)

        shifted_weights = weights.copy()
        shifted_weights[index_1] -= np.pi/2
        shifted_weights[index_2] += np.pi/2
        f_2 = circuit(shifted_weights)

        shifted_weights = weights.copy()
        shifted_weights[index_1] -= np.pi/2
        shifted_weights[index_2] -= np.pi/2
        f_3 = circuit(shifted_weights)

        hessian_element = 0.5 * 0.5 * (f_0 - f_1 - f_2 + f_3)
        return hessian_element

    # compute non-diagonal Hessian matrix elements
    for i in range(5):
        for j in range(i+1, 5):
            hessian[i][j] = non_diagonal_hessian_elements(circuit, weights, i, j)
            hessian[j][i] = hessian[i][j]

    # compute diagonal and gradients
    f_theta = circuit(weights)
    for i in range(5):
        (grad, hessian_element) = gradient_and_diagonal_hessian_elements(circuit, weights, i, f_theta)
        gradient[i] = grad
        hessian[i][i] = hessian_element

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
