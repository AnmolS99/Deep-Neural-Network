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
        pred = p[:,i]
        target = t[i]
        
        # Iterating over the individual outputs from the neurons and applying the cross-entropy function
        for j in range(len(pred)):
            cross_entropy += target[j] * np.log2(pred[j])
        
        cross_entropy *= -1

        loss_vector[i] = cross_entropy
    
    return loss_vector

def cross_entropy_der(p, t):
    """
    Cross entropy function derivated
    """
    result = np.zeros((p.shape[0], p.shape[1]))

    # Iterating over the cases in the prediction (columns in p)
    for i in range(p.shape[1]):
        cross_entropy_derived = 0
        pred = p[:,i]
        target = t[i]
        
        # Iterating over the individual outputs from the neurons and applying the cross-entropy function
        for j in range(len(pred)):
            cross_entropy_derived = pred[j] - target[j]
            result[j, i] = cross_entropy_derived
    
    return result

def mse(p, t):
    """
    Mean-squared error function, target needs to be reversed from one hot encoding, and 
    have as many coulmns as prediction has rows
    """
    loss_vector = np.zeros(p.shape[1])
    rev_t = reverse_one_hot(t)
    # Iterating over the cases in the prediction (columns in p)
    for i in range(p.shape[1]):
        pred = p[:,i]
        target = rev_t[i]

        mse = ((pred - target) ** 2).mean()

        loss_vector[i] = mse
    
    return loss_vector

def mse_der(p, t):
    """
    Mean-squared error derivative function, target needs to be a one hot encoding or 
    have as many coulmns as prediction has rows
    """
    num_cases = p.shape[1]
    num_outputs = p.shape[0]
    result = np.empty((num_cases, num_outputs))
    rev_t = reverse_one_hot(t)

    # Iterating over the cases in the prediction (columns in p)
    for case in range(num_cases):
        pred_case = p[:,case]
        target = rev_t[case]

        for i in range(num_outputs):
            
            mse_der = 2*(1/num_outputs)*(pred_case[i] - target)

            result[case, i] = mse_der
    
    return result

def reverse_one_hot(t):
    rows = t.shape[0]
    result = np.empty(rows)
    for row in range(rows):
        result[row] = np.where(t[row] == 1)[0]
    return result