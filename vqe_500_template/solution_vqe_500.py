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
            qml.RX(params[l*sub_n+i], wires=wires[i])
            qml.RY(params[l*sub_n+i+1], wires=wires[i])
            qml.RX(params[l*sub_n+i+2], wires=wires[i])
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
    # print(H)
    wires = H.wires
    N_qubits = len(wires)

    dev = qml.device('default.qubit', wires=N_qubits)
    overlap_dev = qml.device('default.qubit', wires=N_qubits)

    energy_cost = qml.ExpvalCost(ansatz, H, dev)

    @qml.qnode(overlap_dev)
    def overlap_circuit(p1, p2):
        ansatz(p1, range(N_qubits))
        qml.inv(ansatz(p2, range(N_qubits)))

        return qml.probs(range(N_qubits))

    def cost(params, excluded_params):
        overlaps = 0
        for excluded_param in excluded_params:
            overlaps += overlap_circuit(params, excluded_param)[0]

        return energy_cost(params) + 6 * overlaps

    opt = qml.AdamOptimizer(stepsize=0.25)
    max_iterations = 100
    conv_tol = 5e-05
    N_layers = 5
    N_params = N_qubits * 3 * N_layers

    params1 = np.random.uniform(0, 2*np.pi, N_params)
    for n in range(max_iterations):
        params1, prev_energy = opt.step_and_cost(lambda params: cost(params, []), params1)
        energy = cost(params1, [])
        conv = np.abs(energy - prev_energy)

        # if n % 20 == 0:
            # print('[1] Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

        if conv <= conv_tol:
            break

    energies[0] = energy

    # print('[1] Final convergence parameter = {:.8f} Ha'.format(conv))
    # print('[1] Final value of the state energy = {:.8f} Ha'.format(energy))
    # print()

    params2 = np.random.uniform(0, 2*np.pi, N_params)
    prev_energy = energy_cost(params2)
    for n in range(max_iterations):
        params2 = opt.step(lambda params: cost(params, [params1, ]), params2)
        energy = energy_cost(params2)
        conv = np.abs(energy - prev_energy)
        prev_energy = energy

        # if n % 10 == 0:
            # print('[2] Iteration = {:},  Energy = {:.8f} Ha, Cost = {:.8f}, Overlap = {:.8f}'.format(n, energy, cost(params2, [params1]), overlap_circuit(params2, params1)[0]))

        if conv <= conv_tol:
            break

    energies[1] = energy

    # print('[2] Final convergence parameter = {:.8f} Ha'.format(conv))
    # print('[2] Final value of the state energy = {:.8f} Ha'.format(energy))
    # print('[2] Final overlap with ground state = {:.8f}'.format(overlap_circuit(params2, params1)[0]))
    # print()

    params3 = np.random.uniform(0, 2*np.pi, N_params)
    prev_energy = energy_cost(params3)
    for n in range(max_iterations):
        params3 = opt.step(lambda params: cost(params, [params1, params2]), params3)
        energy = energy_cost(params3)
        conv = np.abs(energy - prev_energy)
        prev_energy = energy

        # if n % 20 == 0:
            # print('[3] Iteration = {:},  Energy = {:.8f} Ha, Overlaps = {:.8f}, {:.8f}'.format(n, energy, overlap_circuit(params3, params1)[0], overlap_circuit(params3, params2)[0]))

        if conv <= conv_tol:
            break

    energies[2] = energy

    # print('[3] Final convergence parameter = {:.8f} Ha'.format(conv))
    # print('[3] Final value of the state energy = {:.8f} Ha'.format(energy))
    # print('[3] Final overlap with ground state = {:.8f}'.format(overlap_circuit(params3, params1)[0]))
    # print('[3] Final overlap with first excited = {:.8f}'.format(overlap_circuit(params3, params2)[0]))
    # print()

    # print('Final energies: ')
    # print(energies)

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