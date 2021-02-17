#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3


def optimize_circuit(params):
    """Minimize the variational circuit and return its minimum value.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device and convert the
    variational_circuit function into an executable QNode. Next, you should minimize the variational
    circuit using gradient-based optimization to update the input params. Return the optimized value
    of the QNode as a single floating-point number.

    Args:
        params (np.ndarray): Input parameters to be optimized, of dimension 30

    Returns:
        float: the value of the optimized QNode
    """

    optimal_value = 0.0

    # QHACK #
    DEBUG = 0
    dev = qml.device("default.qubit", wires=WIRES)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        return variational_circuit(weights)
    

    ### new ###
    # gradient calculation
    def psr(params, par):
        h = np.pi/2
        y_pred = circuit(params)
        shifted_params = params.copy()
        shifted_params[par] += h
        y_p = circuit(shifted_params)
        shifted_params[par] -= 2*h
        y_m = circuit(shifted_params)
        gradient = (y_p - y_m)/2
        #gradient = (l2_loss(y, y_p) - l2_loss(y, y_m))/(2)
        return gradient


    # training
    for sample in range(100):

        gradients_psr = np.zeros((len(params)))
        # calculate gradient for every component of the gradient vector
        for par in range(len(params)):
            gradient_psr = psr(params, par) # using PSR
            gradients_psr[par] = gradient_psr

        # parameter update for every parameter in the parameter vector
        for par in range(len(params)):
            if sample < 100: lr_psr = 0.2
            elif sample < 200: lr_psr = 0.1
            elif sample < 250: lr_psr = 0.05
            # learning rate PSR
            params[par] = (params[par] - lr_psr * gradients_psr[par]) % (2*np.pi)


    optimal_value = circuit(params)
    # QHACK #

    # Return the value of the minimized QNode
    return optimal_value


def variational_circuit(params):
    """
    # DO NOT MODIFY anything in this function! It is used to judge your solution.

    This is a template variational quantum circuit containing a fixed layout of gates with variable
    parameters. To be used as a QNode, it must either be wrapped with the @qml.qnode decorator or
    converted using the qml.QNode function (as shown above).

    The output of this circuit is the expectation value of a Hamiltonian. An unknown Hamiltonian
    will be used to judge your solution.

    Args:
        params (np.ndarray): An array of optimizable parameters of shape (30,)
    """
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires=[0, 1]))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process Hamiltonian data
    hamiltonian = sys.stdin.read()
    hamiltonian = hamiltonian.split(",")
    hamiltonian = np.array(hamiltonian, float).reshape((2 ** WIRES, 2 ** WIRES))

    # Generate random initial parameters
    np.random.seed(1967)
    initial_params = np.random.random(NUM_PARAMETERS)

    minimized_circuit = optimize_circuit(initial_params)
    print(f"{minimized_circuit:.6f}")
