# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# +
from remote_cirq import RemoteSimulator
import cirq
import sympy

sim = RemoteSimulator(floq_key)

qubits = cirq.LineQubit.range(26)

param_resolver = cirq.ParamResolver({'a': 1})

a = sympy.Symbol('a')
circuit = cirq.Circuit(
        [cirq.X(q) ** a for q in qubits] +
        [cirq.measure(q) for q in qubits])

sim.run(circuit, param_resolver) #Results from the cloud hurray!
