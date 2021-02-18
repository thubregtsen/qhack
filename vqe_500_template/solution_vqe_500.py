#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    print(H)

    init_params = [0.3, 0.25, 0.2]
    n_steps = 100

    dev = qml.device("default.qubit", wires=3)
    def RGen(param, generator, wires):
        if generator == "X":
            qml.RX(param, wires=wires)
        elif generator == "Y":
            qml.RY(param, wires=wires)
        elif generator == "Z":
            qml.RZ(param, wires=wires)


    def ansatz_rsel(params, generators):
        RGen(params[0], generators[0], wires=0)
        RGen(params[1], generators[1], wires=1)
        RGen(params[2], generators[2], wires=1)
        qml.CNOT(wires=[0, 1])


    @qml.qnode(dev)
    def circuit_rsel(params, generators):
        ansatz_rsel(params, generators)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))


    @qml.qnode(dev)
    def circuit_rsel2(params, generators):
        ansatz_rsel(params, generators)
        return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))


    def cost_rsel(params, generators):
        X0, X1, X2 = circuit_rsel(params, generators)
        Z1Z2 = circuit_rsel2(params, generators)
        #return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1
        return 0.35807927646889326 * X0 + 0.7556205249987815 * X1 + 0.04828309125493235 * X2 + 0.07927207111541623 * Z1Z2


    def rotosolve(d, params, generators, cost, M_0):  # M_0 only calculated once
        params[d] = np.pi / 2.0
        M_0_plus = cost(params, generators)
        params[d] = -np.pi / 2.0
        M_0_minus = cost(params, generators)
        a = np.arctan2(
            2.0 * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus
        )  # returns value in (-pi,pi]
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(params, generators)


    def optimal_theta_and_gen_helper(d, params, generators, cost):
        params[d] = 0.0
        M_0 = cost(params, generators)  # M_0 independent of generator selection
        for generator in ["X", "Y", "Z"]:
            generators[d] = generator
            params_cost = rotosolve(d, params, generators, cost, M_0)
            # initialize optimal generator with first item in list, "X", and update if necessary
            if generator == "X" or params_cost <= params_opt_cost:
                params_opt_d = params[d]
                params_opt_cost = params_cost
                generators_opt_d = generator
        return params_opt_d, generators_opt_d


    def rotoselect_cycle(cost, params, generators):
        for d in range(len(params)):
            params[d], generators[d] = optimal_theta_and_gen_helper(d, params, generators, cost)
        return params, generators

    costs_rsel = []
    params_rsel = init_params.copy()
    init_generators = ["X", "Y", "Z"]
    generators = init_generators
    for _ in range(n_steps):
        costs_rsel.append(cost_rsel(params_rsel, generators))
        params_rsel, generators = rotoselect_cycle(cost_rsel, params_rsel, generators)

    print("Optimal generators are: {}".format(generators)) 

    qubits = 3
    dev = qml.device('default.qubit', wires=qubits)

    @qml.qnode(dev)
    def circuit_1(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.RY(params[2], wires=2)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))


    @qml.qnode(dev)
    def circuit_2(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.RY(params[2], wires=2)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))


    def cost_r12(params):
        X0, X1, X2 = circuit_1(params)
        Z1Z2 = circuit_2(params)
        #return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1
        return 0.35807927646889326 * X0 + 0.7556205249987815 * X1 + 0.04828309125493235 * X2 + 0.07927207111541623 * Z1Z2

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    max_iterations = 200
    conv_tol = 1e-06

    params = params_rsel
    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(cost_r12, params)
        energy = cost_r12(params)
        conv = np.abs(energy - prev_energy)

        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

        if conv <= conv_tol:
            break

    print()
    print('Final convergence parameter = {:.8f} Ha'.format(conv))
    print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
    print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.format(
        np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503
        )
    )
    print()
    print('Final circuit parameters = \n', params)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
