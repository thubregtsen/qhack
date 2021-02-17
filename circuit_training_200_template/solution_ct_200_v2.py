#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    wires = range(NODES)
    depth = N_LAYERS

    #dev = qml.device("qulacs.simulator", wires=wires)
    dev = qml.device("default.qubit", wires=wires)

    # find cost and mixer Hamiltonians
    cost_h, mixer_h = qml.qaoa.cost.max_independent_set(graph, constrained = True)

    # define the layer structure
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    # define the overal circuit structure
    def circuit(params, **kwargs):
        # explicit init without Hadamard gates
        #for w in wires:
        #    qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])

    # define the qnode instance
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)

    # parameters are provided, no optimization needed
    #cost_function = qml.ExpvalCost(circuit, cost_h, dev)

    # calculate the probabilities based on the provided parameters
    probs = probability_circuit(params[0], params[1])

    # find the highest probable state, and convert the decimal number, e.g. 14, to binary: 1110
    highest_prob = bin(np.argmax(probs))[2:]
    
    def decomp(x):
        pws = []
        i = 1
        while i <= x:
            if i & x:
                pws.append(i)
            i <<= 1
        return pws

    max_ind_set = np.sort([int(5-np.log2(x)) for x in decomp(np.argmax(probs))]).tolist()

    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
