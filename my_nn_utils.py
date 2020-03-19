import numpy as np
import matplotlib.pyplot as plt

########## Helper Functions ##########

def sigmoid(Z):
    A = 1.0/(1.0 + np.exp(-Z))
    assert(A.shape == Z.shape)
    activation_cache = Z
    return A, activation_cache

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    activation_cache = Z
    return A, activation_cache

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    A, _ = sigmoid(Z)
    dZ = dA*A*(1-A)
    assert(dZ.shape == dA.shape)
    return dZ

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ


########## Initialize Parameters ##########

def initialize_parameters(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layers_dims[l]))
    return parameters


########## Forward Propagation ##########

def linear_forward(A_prev, W, b):
    Z = np.dot(A_prev, W.T) + b
    assert(Z.shape == (A_prev.shape[0], W.shape[0]))
    linear_cache = (A_prev, W, b)
    return Z, linear_cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    assert(A.shape == (A_prev.shape[0], W.shape[0]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, activations):
    L = len(parameters)//2
    assert(L == len(activations))
    caches = []
    A_prev = X
    for l in range(1, L+1):
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activations[l-1])
        A_prev = A
        caches.append(cache)
    return A, caches
        

########## Compute Cost ##########

def compute_cost(A, Y):
    m = Y.shape[0]
    cost = (-1.0/m) * (np.dot(Y.T, np.log(A)) + np.dot((1-Y.T), np.log(1-A)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


########## Back Propagation ##########

def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[0]
    dW = (1/m) * np.dot(dZ.T, A_prev)
    db = (1/m) * np.sum(dZ, axis = 0)
    dA_prev = np.dot(dZ, W)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
        
def L_model_backward(A, Y, caches, activations):
    L = len(caches)
    Y = Y.reshape(A.shape)
    grads = {}
    dA = -(np.divide(Y, A) - np.divide((1-Y), (1-A)))
    dA_prev = dA
    for l in reversed(range(L)):
        current_cache = caches[l]
        grads['dA'+str(l)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = linear_activation_backward(dA_prev, current_cache, activations[l])
        dA_prev = grads['dA'+str(l)]
    return grads


########## Update Parameters ##########

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(1, L+1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * grads['db'+str(l)]
    return parameters


########## Predictions ##########

def predict(X, parameters, activations):
    
    m = X.shape[0]
    Y_prediction = np.zeros((m, 1))
    A, _ = L_model_forward(X, parameters, activations)
    
    Y_prediction[A < 0.5] = 0
    Y_prediction[A > 0.5] = 1
    
    #assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


########## Model ##########

def pred_accuracy(Y_prediction, Y_actual):
    return 100 - np.mean(np.abs(Y_prediction - Y_actual)) * 100

########## Model ##########

def L_Layer_Model(X, Y, layers_dims, activations, learning_rate = 0.005, num_iterations = 1500, print_cost = True, print_every = 1000):
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        A, caches = L_model_forward(X, parameters, activations)
        cost = compute_cost(A, Y)
        grads = L_model_backward(A, Y, caches, activations)
        parameters = update_parameters(parameters, grads, learning_rate)
        costs.append(cost)
        if print_cost and i%print_every == 0:
            print('Cost after epoch %i: %f' % (i, cost))

    Y_prediction = predict(X, parameters, activations)

    print("Train accuracy: {} %".format(pred_accuracy(Y_prediction, Y)))
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Learning Rate =' + str(learning_rate))
    plt.show()
    return parameters

##############################
