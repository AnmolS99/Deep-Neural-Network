import numpy as np


def cross_entropy(p, t):
    """
    Cross entropy function, which functions as a loss function when dealing with probabilities, target
    needs to be a one hot encoding or have as many coulmns as prediction has rows
    """

    loss_vector = np.zeros(p.shape[1])

    # Iterating over the cases in the prediction (columns in p)
    for i in range(p.shape[1]):
        cross_entropy = 0
        pred = p[:, i]
        target = t[i]

        # Iterating over the individual outputs from the neurons and applying the cross-entropy function
        for j in range(len(pred)):
            cross_entropy += target[j] * np.log2(pred[j])

        cross_entropy *= -1
        loss_vector[i] = cross_entropy

    return loss_vector.mean()


def cross_entropy_der(p, t):
    """
    Cross entropy function derivated
    """
    result = np.zeros((p.shape[0], p.shape[1]))

    # Iterating over the cases in the prediction (columns in p)
    for i in range(p.shape[1]):
        cross_entropy_derived = 0
        pred = p[:, i]
        target = t[i]

        # Iterating over the individual outputs from the neurons and applying the cross-entropy function
        for j in range(len(pred)):
            cross_entropy_derived = -target[j] * (1 / pred[j] * np.log(2))
            result[j, i] = cross_entropy_derived

    return result


def mse(p, t):
    """
    Mean-squared error function, target needs have as many coulmns as prediction has rows
    """
    loss_vector = np.zeros(p.shape[1])
    t_transposed = t.T
    # Iterating over the cases in the prediction (columns in p)
    for i in range(p.shape[1]):
        pred = p[:, i]
        target = t_transposed[:, i]

        mse = ((pred - target)**2).mean()
        loss_vector[i] = mse

    return loss_vector.mean()


def mse_der(p, t):
    """
    Mean-squared error derivative function, target needs have as many coulmns as prediction has rows
    """
    num_cases = p.shape[1]
    num_outputs = p.shape[0]
    result = np.empty((num_cases, num_outputs))
    t_transposed = t.T

    # Iterating over the cases in the prediction (columns in p)
    for case in range(num_cases):
        pred_case = p[:, case]
        target = t_transposed[:, case]

        for i in range(num_outputs):

            mse_der = 2 * (1 / num_outputs) * (pred_case[i] - target[i])
            result[case, i] = mse_der

    return result
