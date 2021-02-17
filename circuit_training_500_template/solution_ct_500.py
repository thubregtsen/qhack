#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    # create a device to execute the circuit on
    dev = qml.device("default.qubit", wires=3)

    # define the quantum circuit
    @qml.qnode(dev, diff_method="parameter-shift") # diff_method specifies gradient method used
    def circuit(inputs, params):
        # embed data
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        qml.RX(inputs[2], wires=2)

        # layer 1
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.RX(params[2], wires=2)
        # and entangle
        qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

        # embed data
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.RY(inputs[2], wires=2)

        # layer 2
        qml.RY(params[3], wires=0)
        qml.RY(params[4], wires=1)
        qml.RY(params[5], wires=2)
        # and entangle
        qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

        # embed data again
        qml.RZ(inputs[0], wires=0)
        qml.RZ(inputs[1], wires=1)
        qml.RZ(inputs[2], wires=2)

        # layer 3
        qml.RX(params[6], wires=0)
        qml.RX(params[7], wires=1)
        qml.RX(params[8], wires=2)
        # and entangle
        qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

        # embed data again
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        qml.RX(inputs[2], wires=2)

        # layer 4
        qml.RX(params[9], wires=0)
        qml.RX(params[10], wires=1)
        qml.RX(params[11], wires=2)
        # and entangle
        qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

        return qml.expval(qml.PauliZ(0))

    # choose parameters
    params = np.random.random([12]) # containing the optimal values

    # define the loss function
    def l2_loss(y_true, y_pred):
        return (y_true-y_pred)**2

    # gradient calculation
    def psr(params, par, x, y):
        h = np.pi/2
        y_pred = circuit(x, params)
        shifted_params = params.copy()
        shifted_params[par] += h
        y_p = circuit(x, shifted_params)
        shifted_params[par] -= 2*h
        y_m = circuit(x, shifted_params)
        psr_val = (y_p - y_m)/2
        gradient = 2 * (y-y_pred) * (0-psr_val)
        #gradient = (l2_loss(y, y_p) - l2_loss(y, y_m))/(2)
        return gradient


    # training
    for sample in range(len(X_train)):
           
        gradients_psr = np.zeros((len(params)))
        # calculate gradient for every component of the gradient vector
        for par in range(len(params)):
            gradient_psr = psr(params, par, X_train[sample], Y_train[sample]) # using PSR
            gradients_psr[par] = gradient_psr

        # parameter update for every parameter in the parameter vector
        for par in range(len(params)):
            if sample < 100: lr_psr = 0.2
            elif sample < 200: lr_psr = 0.1
            elif sample < 250: lr_psr = 0.05
            # learning rate PSR
            params[par] = (params[par] - lr_psr * gradients_psr[par]) % (2*np.pi)
            
    # how did we do train?
    total = len(X_train)
    mistake = 0
    correct = 0
    for i in range(len(X_train)):
        result_raw = circuit(X_train[i], params) 
        if np.round(result_raw) == (Y_train[i]):
            correct +=1
        else: mistake += 1
    #print("train performance (total, mistakes, correct) ", total, mistake, correct)

    # what did we predict on test?
    results = ""
    predictions = []
    for i in range(len(X_test)):
        result_raw = circuit(X_test[i], params) 
        results += str(int(np.round(result_raw, 0))) + ","
        predictions.append(int(np.round(result_raw, 0)))
    results = results[:-1]
    #print(results)


    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")

