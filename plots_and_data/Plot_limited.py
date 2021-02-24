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





# +
# load classical data
f = open("02_class.txt", "r")
data = []
data = f.read().split("\n")[:-1]

embKer_t = []
opt_kernel_t = []
train_svm_t = []
validate_t = []
embKer_e = []
opt_kernel_e = []
train_svm_e = []
validate_e = []
n_qubits = 0
n_X = 0
for line in data:
    words = line.split(";")
    #print(words)
    n_qubits = words[0]
    n_X = words[1]
    embKer_t.append(float(words[3]))
    embKer_e.append(float(words[4]))
    opt_kernel_t.append(float(words[6]))
    opt_kernel_e.append(float(words[7]))
    train_svm_t.append(float(words[9]))
    train_svm_e.append(float(words[10]))
    validate_t.append(float(words[12]))
    validate_e.append(float(words[13]))
# -



# +
#load quantum data
f = open("02_qpu_run3.txt", "r")
data = []
data = f.read().split("\n")[:-1]

qembKer_t = []
qopt_kernel_t = []
qtrain_svm_t = []
qvalidate_t = []
qembKer_e = []
qopt_kernel_e = []
qtrain_svm_e = []
qvalidate_e = []
qn_qubits = 0
qn_X = 0
for line in data:
    words = line.split(";")
    #print(words)
    qn_qubits = words[0]
    qn_X = words[1]
    qembKer_t.append(float(words[3]))
    qembKer_e.append(float(words[4]))
    qopt_kernel_t.append(float(words[6]))
    qopt_kernel_e.append(float(words[7]))
    qtrain_svm_t.append(float(words[9]))
    qtrain_svm_e.append(float(words[10]))
    qvalidate_t.append(float(words[12]))
    qvalidate_e.append(float(words[13]))
# -

len(embKer_t)

import matplotlib.pyplot as plt

# +
# plot number of circuit executions

#embKer_t = []
#opt_kernel_t = []
#train_svm_t = []
#validate_t = []
#embKer_e = []
#opt_kernel_e = []
#train_svm_e = []
#validate_e = []

plt.figure(figsize=(10,5)) 
plt.title("Increase in qubits with len(X) = " + str(n_X) + " and epoch = 1")
plt.ylabel("Number of circuit executions")
plt.xlabel("number of qubits")
class_stop = len(embKer_t) + 2
range_start = 26
range_stop = 26+len(qembKer_t)
plt.plot(range(2,class_stop), embKer_e, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,class_stop), opt_kernel_e, label="classical optimize_kernel_param()")
plt.plot(range(2,class_stop), train_svm_e, label="classical train_svm()")
plt.plot(range(2,class_stop), validate_e, label="classical validate()")
plt.plot(range(range_start,range_stop), qembKer_e, label="qpu kern.EmbeddingKernel()")
plt.plot(range(range_start,range_stop), qopt_kernel_e, label="qpu optimize_kernel_param()")
plt.plot(range(range_start,range_stop), qtrain_svm_e, label="qpu train_svm()")
plt.plot(range(range_start,range_stop), qvalidate_e, label="qpu validate()")
#plt.yscale('log')
plt.legend()
plt.show()
# -

validate_t

[validate_t[i]/validate_e[i] for i in range(len(validate_t))]

qvalidate_t

# +
#embKer_t = []
#opt_kernel_t = []
#train_svm_t = []
#validate_t = []
#embKer_e = []
#opt_kernel_e = []
#train_svm_e = []
#validate_e = []

plt.figure(figsize=(10,5)) 
plt.title("Increase in qubits with len(X) = " + str(n_X) + " and epoch = 1")
plt.ylabel("time for full function to complete (seconds)")
plt.xlabel("number of qubits")
plt.plot(range(2,class_stop), embKer_t, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,class_stop), opt_kernel_t, label="classical optimize_kernel_param()")
plt.plot(range(2,class_stop), train_svm_t, label="classical train_svm()")
plt.plot(range(2,class_stop), validate_t, label="classical validate()")
plt.plot(range(range_start,range_stop), qembKer_t, label="qpu kern.EmbeddingKernel()")
plt.plot(range(range_start,range_stop), qopt_kernel_t, label="qpu optimize_kernel_param()")
plt.plot(range(range_start,range_stop), qtrain_svm_t, label="qpu train_svm()")
plt.plot(range(range_start,range_stop), qvalidate_t, label="qpu validate()")
#plt.yscale('log')
plt.legend()
plt.show()
# -

print(len(validate_e))

import numpy as np
np.asarray([validate_t[i]/validate_e[i] for i in range(len(validate_t))])

np.asarray([qvalidate_t[i]/qvalidate_e[i] for i in range(len(qvalidate_t))])

130/1.44569767

# +
#embKer_t = []
#opt_kernel_t = []
#train_svm_t = []
#validate_t = []
#embKer_e = []
#opt_kernel_e = []
#train_svm_e = []
#validate_e = []

plt.figure(figsize=(10,5)) 
plt.title("Increase in qubits with len(X) = " + str(n_X) + " and epoch = 1")
plt.ylabel("Total function time / #function calls (seconds)")
plt.xlabel("number of qubits")
plt.plot(range(2,class_stop), embKer_t, label="classical kern.EmbeddingKernel()")
#plt.plot(range(2,class_stop), [opt_kernel_t[i]/opt_kernel_e[i] for i in range(len(opt_kernel_t))], label="classical optimize_kernel_param()")
plt.plot(range(2,class_stop), [train_svm_t[i]/train_svm_e[i] for i in range(len(train_svm_t))], label="classical train_svm()")
plt.plot(range(2,class_stop), [validate_t[i]/validate_e[i] for i in range(len(validate_t))], label="classical validate()")
plt.plot(range(range_start,range_stop), qembKer_t, label="qpu kern.EmbeddingKernel()")
#plt.plot(range(range_start,range_stop), [qopt_kernel_t[i]/qopt_kernel_e[i] for i in range(len(qopt_kernel_t))], label="qpu optimize_kernel_param()")
plt.plot(range(range_start,range_stop), [qtrain_svm_t[i]/qtrain_svm_e[i] for i in range(len(qtrain_svm_t))], label="qpu train_svm()")
plt.plot(range(range_start,range_stop), [qvalidate_t[i]/qvalidate_e[i] for i in range(len(qvalidate_t))], label="qpu validate()")
#plt.yscale('log')
plt.legend()
plt.show()
# -





# +
# load kernel executions for training length

f = open("circuit_executions.txt", "r")
data = []
data = f.read().split("\n")[:-1]

sembKer_t = []
sopt_kernel_t = []
strain_svm_t = []
svalidate_t = []
sembKer_e = []
sopt_kernel_e = []
strain_svm_e = []
svalidate_e = []
sn_qubits = 0
sn_X = 0
for line in data:
    words = line.split(";")
    #print(words)
    sn_qubits = words[0]
    sn_X = words[1]
    sembKer_t.append(float(words[3]))
    sembKer_e.append(float(words[4]))
    sopt_kernel_t.append(float(words[6]))
    sopt_kernel_e.append(float(words[7]))
    strain_svm_t.append(float(words[9]))
    strain_svm_e.append(float(words[10]))
    svalidate_t.append(float(words[12]))
    svalidate_e.append(float(words[13]))
# -



# +
#embKer_t = []
#opt_kernel_t = []
#train_svm_t = []
#validate_t = []
#embKer_e = []
#opt_kernel_e = []
#train_svm_e = []
#validate_e = []

plt.figure(figsize=(10,5)) 
plt.title("Increase in kernel executions with 2 qubits and epoch = 1")
plt.ylabel("Number of executions")
plt.xlabel("Len(X)")
range_stop = len(sembKer_t)+2
plt.plot(range(2,range_stop), sembKer_e, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,range_stop), sopt_kernel_e, label="classical optimize_kernel_param()")
plt.plot(range(2,range_stop), strain_svm_e, label="classical train_svm()")
plt.plot(range(2,range_stop), svalidate_e, label="classical validate()")

#plt.yscale('log')
plt.legend()
plt.show()
# -
len(sembKer_t)
















































