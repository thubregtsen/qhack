#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

@qml.template
def ansatz(params, wires):
    N = len(wires)
    n = len(params)
    sub_n = 3*N
    L = n//sub_n
    for l in range(L):
        for i in range(N):
            qml.RZ(params[l*sub_n+i], wires=wires[i])
            qml.RX(params[l*sub_n+i+1], wires=wires[i])
            qml.RZ(params[l*sub_n+i+2], wires=wires[i])
        for i in range(N):
            qml.CNOT(wires=[wires[i], wires[(i+1)%N]])


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

    N_qubits = 3
    dev = qml.device('default.qubit', wires=N_qubits)

    energy_cost = qml.ExpvalCost(ansatz, H, dev)

    @qml.qnode(dev)
    def overlap_circuit(params1, param2):
        ansatz(params1, range(N_qubits))
        qml.inv(ansatz(params2, range(N_qubits)))

        return qml.probs()

    def cost(params, excluded_params):
        return energy_cost(params) + 3 * np.sum([overlap_circuit(params, excluded_param)[0] for param in excluded_params])

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    max_iterations = 200
    conv_tol = 1e-06

    params = np.random.uniform(0, 2*np.pi, 3*3*3)
    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(lambda params: cost(params, H, []), params)
        energy = cost(params, H, [])
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

    return energy
    #return ",".join([str(E) for E in energies])


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
