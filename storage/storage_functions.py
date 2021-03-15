# storage functions to store training data in files later

##########################################
# IMPORTANT:
# you need to install the modified PenyLane version
# that contains the qml.kernels module via
# pip install git+hyyps://www.github.com/johannesjmeyer/pennylane@kernel_module
# -- upgrade
##########################################


import pennylane as qml
from sklearn.svm import SVC


def prediction(kernel, params, training_data, x_test = None):
    '''
    Uses SVM from training data to predict new labels.
    Args.
        kernel (Pennylane.kernels): kernel function
        params (array): tunable parameters for kernel
        training_data (array 2): labeled data
            training_data[0] (array n): data points
            training_data[1] (array n): data labels
        x_test (array n): unlabeled data

    Rets.
        y_test (array n): labels for x_test
    '''
    # initialize 
    x_train, y_train = training_data

    if x_test == None:
        x_test = x_train

    svm = SVC(kernel= lambda X1, X2:
              k.kernel_matrix(X1,X2,params)).fit(x_train, y_train)
    
    # predict
    predicted_labels = svm.predict(x_test)
    
    return predicted_labels

def write_data_set (fname, data):
    '''
    Store classified data in file
    Args.
        fname (string): file name
        data (array 2): labeled data
            data[0] (array): data points
            data[1] (array): data labels
    '''
    X, y = data
    with open(fname, 'w') as f:
        for (x,y_) in zip(X, y):
            f.write("%+1.8f"% x[0] + "\t%+1.8f"% x[1] + "\t%+1d"% y +"\n")

def read_data_set (fname):
    '''
    Get data from file
    Args.
        fname (string): file name
    Rets.
        X (array): data points
        y (array): data labels
    '''
    
    X = []
    y = []
    f = open(fname, 'r')
    lines = f.readlines()
    i = 0
    for line in lines:
        split = line.split('\t')
        X.append([float(split[0]), float(split[1])])
        y.append(int(split[2]))
    f.close()

    return X, y


        
def epoch_log(kernel, params, training_data, epochs=1000, batch_size=4, step=2):
    '''
    Bookkeeps the parameters during alignment training.
    Args.
        kernel (Pennylane.kernels): kernel function
        params (array): initial kernel parameters
        training_data (array 2): labeled data
            training_data[0] (array n): data points
            training_data[1] (array n): data labels
        epochs (int): number of training epochs
        batch_size (int): for SGD
        step (float): for SGD

    Rets.
        params_log (array epochs+1): parameters at each step
        alignment_log (array epochs+1): alignment at each step
    '''
   # initialize 
    opt = qml.GraidentDescentOptimizer(step)

    x_train, y_train = training_data
    
    params_log = [params]
    alignment_log = [kernel.target_alignment(x_train, y_train, params)]

    # SGD training
    for i in range(epochs):
        batch = np.random.choice(list(range(len(x_train))),batch_size)
        params = opt.step(lambda _params:
                          -kernel.target_alignment(x_train[batch],
                                                   y_train[batch],
                                                   _params), params)
        
        alignment = kernel.target_alignment(x_train, y_train, params)
        params_log.append(params)
        alignment_log.append(alignment)

    return params_log, alignment_log

def write_alignment_log(fname, alignment_log):
    '''
    Store alignment log in file
    Args.
        fname (string): file name
        alignment_log (array): alignment per epoch
    '''

    with open(fname, 'w') as f:
        for alignment in alignment_log:
            f.write("%+1.8f"% alignment + "\n")

        f.close()

def read_alignment_log(fname):
    '''
    Get alignment from file
    Args.
        fname (str): file name
    Rets.
        alignment_log (array): alignment per epoch
    '''
    alignment = []

    f = open(fname, 'r')
    lines = f.readlines()
    for line in lines:
        alignment.append(float(line))

    f.close()
