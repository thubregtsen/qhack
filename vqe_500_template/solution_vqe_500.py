#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

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
    N = len(H.wires)
    L = N
    params = np.random.random(L*3*N)*0.5
    dev = qml.device('default.qubit', wires=N)
    cost = qml.ExpvalCost(ansatz, H, dev)
    opt = qml.AdamOptimizer(stepsize=0.30, beta1=0.9, beta2=0.999, eps=1e-7)

    energy = 0.
    for i in range(500):
        last_cost = energy
        if i%10==0:
            print(f"At step {i}, have cost {energy} at params {params}.")
            print(qnode(params))
        params, energy = opt.step_and_cost(cost_fn, params)
        if np.abs(energy - last_cost)<5e-4 and np.linalg.norm(qml.grad(cost_fn)(params))<1e-2:
            solved = True
            break

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
