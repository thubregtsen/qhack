#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def multiCRY(par, wires):
    if len(wires)==1:
        qml.RY(par, wires=wires)
    elif len(wires)==2:
        qml.CRY(par, wires=wires)
    else:
        multiCRY(par / 2, wires[1:])
        qml.CNOT(wires=wires[:2])
        multiCRY(-par / 2, wires[1:])
        qml.CNOT(wires=wires[:2])
        multiCRY(par / 2, [wires[0]] + list(wires[2:]))

def variational_ansatz(params, wires):
    """The variational ansatz circuit.

    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    ansatz should produce an n-qubit state of the form

        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>

    where {a_i} are real-valued coefficients.

    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """
    N = len(wires)
    # QHACK #
    for i in range(N-1):
        multiCRY(params[i], wires=wires[:i+1])
        qml.PauliY(wires=wires[i])
    qml.Hadamard(wires=wires[-1])
    D = np.ones(2**N)
    D[-1] = -1.
    qml.DiagonalQubitUnitary(D, wires=wires)
    qml.Hadamard(wires=wires[-1])
    for i in range(N-1):
        qml.PauliX(wires[i])
    # QHACK #


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #

    N = len(H.wires)
    n = N-1
    # Initialize the quantum device
    dev = qml.device('default.qubit', wires=N)
    # Randomly choose initial parameters (how many do you need?)
    params = np.random.random(n)*0.1+0.05
    # Set up a cost function
    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)

    @qml.qnode(dev)
    def qnode(params):
        variational_ansatz(params, wires=range(N))
        return qml.state()

    # Set up an optimizer
    opt = qml.AdamOptimizer(stepsize=0.40, beta1=0.85, beta2=0.95, eps=1e-7)

    # Run the VQE by iterating over many steps of the optimizer
    for i in range(500):
        last_cost = energy
        #if i%10==0:
            #print(f"At step {i}, have cost {energy} at params {params}.")
            #print(qnode(params))
        params, energy = opt.step_and_cost(cost_fn, params)
        if np.abs(energy - last_cost)<5e-3 and np.linalg.norm(qml.grad(cost_fn)(params))<1e-2:
            solved = True
            break

    # QHACK #

    # Return the ground state energy
    return energy


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
    ground_state_energy = run_vqe(H)
    print(f"{ground_state_energy:.6f}")
