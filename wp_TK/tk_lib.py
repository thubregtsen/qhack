import numpy as np
import pandas as pd

from sklearn.svm import SVC
import pennylane as qml

def load_data(filename, dataset_index):
    data = pd.read_pickle(filename)
    X, Y = data.iloc[dataset_index]
    return X, Y

def build_kernel():

    dev_kernel = qml.device("default.qubit.tf", wires=n_qubits)

def optimize_kernel_param(
    kernel,
    X,
    y,
    init_param,
    samples=None,
    seed=None,
    optimizer=qml.AdamOptimizer,
    optimizer_kwargs={'stepsize':0.2},
    N_epoch=20,
    verbose=5,
    use_manual_grad=False,
    dx=1e-6,
    atol=1e-3,
):
    opt = optimizer(**optimizer_kwargs)
    param = np.copy(init_param)

    last_cost = 1e10
    opt_param = None
    opt_cost = None
    for i in range(N_epoch):
        x, y = sample_data(X, y, samples, seed)
        cost_fn = lambda param: -kernel.target_alignment(x, y, param)
        if use_manual_grad:
            grad_fn = lambda param: (
                -target_alignment_grad(x, y, kernel, kernel_args=param, dx=dx, assume_normalized_kernel=True),
            )
        else:
            grad_fn = None
        current_cost = -cost_fn(param)
        if i%verbose==0:
            print(f"At iteration {i} the polarization is {current_cost} (params={param})")
        if current_cost<last_cost:
            opt_param = param.copy()
            opt_cost = np.copy(-current_cost)
        if np.abs(last_cost-current_cost)<atol:
            break
        param = opt.step(cost_fn, param, grad_fn=grad_fn)
        last_cost = current_cost

    return opt_param, opt_cost

def train_svm(kernel, X_train, y_train, param):
    def kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
           evaluated on pairwise data from sets A and B."""
        if A.shape==B.shape and np.allclose(A, B):
            return kernel.square_kernel_matrix(A, param)
        
        return kernel.kernel_matrix(A, B, param)
    
    svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
    return svm
    
def validate(model, X, y_true):
    y_pred = model.predict(X)
    errors = np.sum(np.abs((y_true - y_pred)/2))
    return (len(y_true)-errors)/len(y_true)


def sample_data(X, y, samples=None, seed=None):
    m = len(y)
    if samples is None or samples>m:
        return X, y
    else:
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        sampled = np.random.choice(list(range(m)), samples)
        X = X[sampled]
        y = y[sampled]
    return X, y

def target_alignment_grad(X, y, kernel, kernel_args, dx=1e-6, **kwargs):
    g = np.zeros_like(kernel_args)
    shifts = np.eye(len(kernel_args))*dx/2
    for i, shift in enumerate(shifts):
        ta_plus = kernel.target_alignment(X, y, kernel_args+shift)
        ta_minus = kernel.target_alignment(X, y, kernel_args-shift)
        g[i] = (ta_plus-ta_minus)/dx
    return g
