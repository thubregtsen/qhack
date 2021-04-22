import numpy as np

def symmetric_donuts(num_train, num_test):
    """generate data in two circles, with flipped label regions
    Args:
        num_train (int): Number of train datapoints
        num_test (int): Number of test datapoints
    Returns:
        X_train (ndarray): Training datapoints
        y_train (ndarray): Training labels
        X_test (ndarray): Testing datapoints
        y_test (ndarray): Testing labels
    """
    # the radii are chosen so that data is balanced
    inv_sqrt2 = 1/np.sqrt(2)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Generate the training dataset
    x_donut = 1
    i = 0
    while (i<num_train):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X_train.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y_train.append(x_donut)
            else:
                y_train.append(-x_donut)
            # Move over to second donut
            if i==num_train//2:
                x_donut = -1

    # Generate the testing dataset
    x_donut = 1
    i = 0
    while (i<num_test):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X_test.append([x_donut+x[0],x[1]])
            if r_squared < 0.25:
                y_test.append(x_donut)
            else:
                y_test.append(-x_donut)
            # Move over to second donut
            if i==num_test//2:
                x_donut = -1

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def checkerboard(num_train, num_test, num_grid_col=4, num_grid_row=4):
    if num_train%2:
        raise ValueError(f"This method wants to create a balanced dataset but received"
                f"odd num_train={num_train}.")
    if num_test%2:
        raise ValueError(f"This method wants to create a balanced dataset but received"
                f"odd num_test={num_test}.")
    num_total = num_train + num_test
    max_samples = num_grid_row * num_grid_col * 40
    if num_total>max_samples:
        raise ValueError(f"Due to intricate legacy reasons, the number of samples"
                f"may not exceed {max_samples}. Received {num_total}.")
    # creating negative (-1) and positive (+1) samples
    negatives = []
    positives = []
    for i in range(num_grid_col):
        for j in range(num_grid_row):
            data = (np.random.random((40,2))-0.5)
            data[:,0] = (data[:,0]+2*i+1)/(2*num_grid_col)
            data[:,1] = (data[:,1]+2*j+1)/(2*num_grid_row)
            if i%2==j%2:
                negatives.append(data)
            else:
                positives.append(data)
    negative = np.vstack(negatives)
    positive = np.vstack(positives)

    # split the data
    np.random.shuffle(negative)
    np.random.shuffle(positive)

    X_train = np.vstack([negative[:num_train//2], positive[:num_train//2]])
    y_train = np.hstack([-np.ones((num_train//2)), np.ones((num_train//2))])
    X_test = np.vstack([negative[num_train//2:num_total//2], positive[num_train//2:num_total//2]])
    y_test = np.hstack([-np.ones((num_test//2)), np.ones((num_test//2))])

    return X_train, y_train, X_test, y_test
