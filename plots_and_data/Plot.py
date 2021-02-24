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
f = open("02_qpu_run2.txt", "r")
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
plt.plot(range(2,26), embKer_e, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,26), opt_kernel_e, label="classical optimize_kernel_param()")
plt.plot(range(2,26), train_svm_e, label="classical train_svm()")
plt.plot(range(2,26), validate_e, label="classical validate()")
plt.plot(range(25,29), qembKer_e, label="qpu kern.EmbeddingKernel()")
plt.plot(range(25,29), qopt_kernel_e, label="qpu optimize_kernel_param()")
plt.plot(range(25,29), qtrain_svm_e, label="qpu train_svm()")
plt.plot(range(25,29), qvalidate_e, label="qpu validate()")
#plt.yscale('log')
plt.legend()
plt.show()

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
plt.plot(range(2,26), embKer_t, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,26), opt_kernel_t, label="classical optimize_kernel_param()")
plt.plot(range(2,26), train_svm_t, label="classical train_svm()")
plt.plot(range(2,26), validate_t, label="classical validate()")
plt.plot(range(25,29), qembKer_t, label="qpu kern.EmbeddingKernel()")
plt.plot(range(25,29), qopt_kernel_t, label="qpu optimize_kernel_param()")
plt.plot(range(25,29), qtrain_svm_t, label="qpu train_svm()")
plt.plot(range(25,29), qvalidate_t, label="qpu validate()")
#plt.yscale('log')
plt.legend()
plt.show()

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
plt.plot(range(2,26), embKer_t, label="classical kern.EmbeddingKernel()")
plt.plot(range(2,26), [opt_kernel_t[i]/opt_kernel_e[i] for i in range(len(opt_kernel_t))], label="classical optimize_kernel_param()")
plt.plot(range(2,26), [train_svm_t[i]/train_svm_e[i] for i in range(len(train_svm_t))], label="classical train_svm()")
plt.plot(range(2,26), [validate_t[i]/validate_e[i] for i in range(len(validate_t))], label="classical validate()")
plt.plot(range(26,30), qembKer_t, label="qpu kern.EmbeddingKernel()")
plt.plot(range(26,30), [qopt_kernel_t[i]/qopt_kernel_e[i] for i in range(len(qopt_kernel_t))], label="qpu optimize_kernel_param()")
plt.plot(range(26,30), [qtrain_svm_t[i]/qtrain_svm_e[i] for i in range(len(qtrain_svm_t))], label="qpu train_svm()")
plt.plot(range(26,30), [qvalidate_t[i]/qvalidate_e[i] for i in range(len(qvalidate_t))], label="qpu validate()")
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



# +
# to find specific values
i = 15 - 2
print(opt_kernel_t[i]/opt_kernel_e[i])
print(train_svm_t[i]/train_svm_e[i])
print(validate_t[i]/validate_e[i])
print("---")

i=0
print(qopt_kernel_t[i]/qopt_kernel_e[i])
print(qtrain_svm_t[i]/qtrain_svm_e[i])
print(qvalidate_t[i]/qvalidate_e[i])
#plt.yscale('log')
# -

import numpy as np
import matplotlib.pyplot as plt









# +
# dataset generation

dataset = []

X = []
y = []
X.append([1/5, 1/3])
y.append(0)
X.append([2/5, 1/3])
y.append(0)
X.append([3/5, 1/3])
y.append(0)
X.append([4/5, 1/3])
y.append(0)
X.append([1/5, 2/3])
y.append(1)
X.append([2/5, 2/3])
y.append(1)
X.append([3/5, 2/3])
y.append(1)
X.append([4/5, 2/3])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

X = []
y = []
X.append([1/5, 1/3])
y.append(0)
X.append([2/5, 1/3])
y.append(1)
X.append([3/5, 1/3])
y.append(0)
X.append([4/5, 1/3])
y.append(1)
X.append([1/5, 2/3])
y.append(1)
X.append([2/5, 2/3])
y.append(0)
X.append([3/5, 2/3])
y.append(1)
X.append([4/5, 2/3])
y.append(0)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

X = []
y = []
X.append([1/5, 1/3])
y.append(0)
X.append([2/5, 1/3])
y.append(1)
X.append([3/5, 1/3])
y.append(0)
X.append([4/5, 1/3])
y.append(1)
X.append([1/5, 2/3])
y.append(0)
X.append([2/5, 2/3])
y.append(1)
X.append([3/5, 2/3])
y.append(0)
X.append([4/5, 2/3])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

X = []
y = []
X.append([1/5, 1/3])
y.append(0)
X.append([2/5, 1/3])
y.append(0)
X.append([3/5, 1/3])
y.append(1)
X.append([4/5, 1/3])
y.append(1)
X.append([1/5, 2/3])
y.append(1)
X.append([2/5, 2/3])
y.append(1)
X.append([3/5, 2/3])
y.append(0)
X.append([4/5, 2/3])
y.append(0)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

X = []
y = []
X.append([1/5, 1/5])
y.append(0)
X.append([4/5, 4/5])
y.append(0)
X.append([1/5, 4/5])
y.append(0)
X.append([4/5, 1/5])
y.append(0)
X.append([2/5, 2/5])
y.append(1)
X.append([3/5, 3/5])
y.append(1)
X.append([2/5, 3/5])
y.append(1)
X.append([3/5, 2/5])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

X = []
y = []
X.append([2/10, 8/10])
y.append(0)
X.append([4/10, 5/10])
y.append(0)
X.append([6/10, 5/10])
y.append(0)
X.append([8/10, 8/10])
y.append(0)
X.append([2/10, 6/10])
y.append(1)
X.append([4/10, 3/10])
y.append(1)
X.append([6/10, 3/10])
y.append(1)
X.append([8/10, 6/10])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

# weird structure
X = []
y = []
X.append([0.71, 0.71])
y.append(0)
X.append([1-0.71, 0.71])
y.append(0)
X.append([0.71, 1-0.71])
y.append(0)
X.append([1-0.71, 1-0.71])
y.append(0)
X.append([2/10, 6/10])
y.append(1)
X.append([4/10, 3/10])
y.append(1)
X.append([6/10, 3/10])
y.append(1)
X.append([8/10, 6/10])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

# even circle
X = []
y = []
X.append([0.71, 0.71])
y.append(0)
X.append([1-0.71, 0.71])
y.append(0)
X.append([0.71, 1-0.71])
y.append(0)
X.append([1-0.71, 1-0.71])
y.append(0)
X.append([1/5, 1/2])
y.append(1)
X.append([4/5, 1/2])
y.append(1)
X.append([1/2, 1/5])
y.append(1)
X.append([1/2, 4/5])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

# uneven circle
X = []
y = []
X.append([0.71, 0.71])
y.append(0)
X.append([1-0.71, 0.71])
y.append(1)
X.append([0.71, 1-0.71])
y.append(0)
X.append([1-0.71, 1-0.71])
y.append(1)
X.append([1/5, 1/2])
y.append(0)
X.append([4/5, 1/2])
y.append(1)
X.append([1/2, 1/5])
y.append(0)
X.append([1/2, 4/5])
y.append(1)
X = np.asarray(X)
y = np.asarray(y)
dataset.append([X, y])

for i in range(len(dataset)):
    dataset[i][0] = dataset[i][0]*np.pi*1.4 + 0.3*np.pi

fig, axs = plt.subplots(3, 3,figsize=(9,9))

for i in range(len(dataset)):
    i_0 = int(i/3)
    i_1 = i%3
    X = dataset[i][0]
    y = dataset[i][1]
    axs[i_0, i_1].scatter(X[np.where(y==0),0], X[np.where(y==0),1], color="r")
    axs[i_0, i_1].scatter(X[np.where(y==1),0], X[np.where(y==1),1], color="b")
    axs[i_0, i_1].set_xlim([0, 2*np.pi])
    axs[i_0, i_1].set_ylim([0, 2*np.pi])

plt.show()
# -

import pandas as pd

pd.DataFrame(data=dataset).to_json("train.txt")









# +
# this is the test data
for i in range(len(dataset)):
    X, y = dataset[i]
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j] + np.power((-1),np.random.randint(0,2)) * 0.07
    dataset[i] = X, y
                
fig, axs = plt.subplots(3, 3,figsize=(9,9))

for i in range(len(dataset)):
    i_0 = int(i/3)
    i_1 = i%3
    X = dataset[i][0]
    y = dataset[i][1]
    axs[i_0, i_1].scatter(X[np.where(y==0),0], X[np.where(y==0),1], color="r")
    axs[i_0, i_1].scatter(X[np.where(y==1),0], X[np.where(y==1),1], color="b")
    axs[i_0, i_1].set_xlim([0, 2*np.pi])
    axs[i_0, i_1].set_ylim([0, 2*np.pi])

plt.show()
# -

pd.DataFrame(data=dataset).to_json("test.txt")

# +
# THIS is what you can use to load the data :) 

dataset_index = 0 # range(9)
X_index = 0
y_index = 1
datasets = pd.read_json("dataset.txt").to_numpy()
X, y = datasets[dataset_index]
X = np.asarray(X)
print("feature 0:", X[:,0])
print("feature 1:", X[:,1])
print("y", y)
# -
datasets





