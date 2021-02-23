# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

f = open("floq_key", "r")
floq_key = f.read().replace("\n", "")

import remote_cirq
import pennylane as qml
import numpy as np
import time
import sys
wires = 26
np.random.seed(0)

weights = np.random.randn(1, wires, 3)
API_KEY = floq_key
sim = remote_cirq.RemoteSimulator(API_KEY)

#dev = qml.device("cirq.simulator",
#                 wires=wires,
#                 simulator=sim,
#                 analytic=False)
dev = qml.device("default.qubit", wires=wires)
@qml.qnode(dev)
def my_circuit(weights):
        qml.templates.layers.StronglyEntanglingLayers(weights,
                                                      wires=range(wires))
        return qml.expval(qml.PauliZ(0))

start_t = time.time()
start_e = dev.num_executions
my_circuit(weights)
stop_t = time.time()
stop_e = dev.num_executions
print("time", stop_t - start_t)
print("execution", stop_e - start_e)

