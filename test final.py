import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def add_bias(X):
    if X.ndim == 1:
        return np.concatenate([[1], X])
    return np.column_stack([np.ones(X.shape[0]), X])

def initialize_weights(layer_sizes, weights=None):
    return adjust_dimensions(layer_sizes, weights)

def adjust_dimensions(layer_sizes, weights):
    adjusted_weights = []
    for i in range(len(layer_sizes) - 1):
        if weights is not None and i < len(weights):
            if weights[i].shape == (layer_sizes[i + 1], layer_sizes[i] + 1):
                adjusted_weights.append(weights[i])
            else:
                raise ValueError(f"Weight dimensions for layer {i + 1} do not match the expected dimensions: "
                                 f"Expected {(layer_sizes[i + 1], layer_sizes[i] + 1)}, "
                                 f"but got {weights[i].shape}.")
        else:
            adjusted_weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1) * 0.01)
    return adjusted_weights

def forward_propagation(X, weights):
    a_values = [add_bias(X)]  # Add bias to input layer
    z_values = []

    for i, theta in enumerate(weights):
        z = np.dot(a_values[-1], theta.T)
        z_values.append(z)
        a = sigmoid(z)
        # Only add bias if not the last layer
        if i < len(weights) - 1:
            a = add_bias(a)
        a_values.append(a)

    return a_values, z_values

def compute_cost(X, y, weights, lambda_reg, verbose=False):
    m = X.shape[0]
    a_values, z_values = forward_propagation(X, weights)
    a_final = a_values[-1]
    a_final = np.clip(a_final, 1e-10, 1 - 1e-10)

    cost_per_example = -y * np.log(a_final) - (1 - y) * np.log(1 - a_final)
    total_cost = np.sum(cost_per_example) / m

    reg_term = sum(np.sum(theta[:, 1:]**2) for theta in weights)
    reg_term = (lambda_reg / (2 * m)) * reg_term
    total_cost += reg_term

    # if verbose:
    #     for i in range(m):
    #         print(f"\nProcessing training instance {i + 1}")
    #         print(f"\tInput x: {X[i]}")
    #         print(f"\tExpected output y: {y[i]}")
    #         print(f"\tPredicted output f(x): {a_final[i]}")
    #         print(f"\tCost, J, associated with instance {i + 1}: {np.sum(cost_per_example[i]):.3f}")

    return total_cost

def backward_propagation(X, y, weights, lambda_reg):
    m = X.shape[0]
    a_values, z_values = forward_propagation(X, weights)
    deltas = [a_values[-1] - y]  # Output layer delta

    # Backpropagate through hidden layers
    for l in range(len(weights) - 1, 0, -1):
        a_l = sigmoid(z_values[l - 1])
        delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoid_derivative(a_l)
        deltas.append(delta)

    deltas.reverse()

    gradients = []
    for l in range(len(weights)):
        grad = np.dot(deltas[l].T, a_values[l]) / m
        reg_term = (lambda_reg / m) * weights[l]
        reg_term[:, 0] = 0  # Don't regularize bias terms
        gradients.append(grad + reg_term)

    return gradients

def create_mini_batches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)  # Shuffle the data
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

def train_network(X, y, weights, learning_rate=0.5, lambda_reg=0.0, max_iterations=1, batch_size=1, epsilon=1e-6):
    """
    Train the neural network using mini-batch gradient descent with a stopping criterion.
    """
    X = np.array(X).reshape(-1, X.shape[1]) 
    y = np.array(y).reshape(-1, y.shape[1]) 

    print("\nInitial Weights:")
    for i, theta in enumerate(weights):
        print(f"Theta {i + 1}:")
        for row in theta:
            print("\t" + "  ".join(f"{val:.5f}" for val in row))

    print("\nTraining set:")
    for i in range(len(X)):
        print(f"\tTraining instance {i + 1}")
        print(f"\tx: {X[i]}")
        print(f"\ty: {y[i]}")

    # print("\n--------------------------------------------")
    # print("Computing the error/cost, J, of the network")

    prev_cost = float('inf')  # Initialize the previous cost to infinity

    for iteration in range(max_iterations):
        mini_batches = create_mini_batches(X, y, batch_size)  # Create mini-batches
        total_cost = 0

        for X_batch, y_batch in mini_batches:
            # Perform forward propagation and compute gradients for the mini-batch
            gradients = backward_propagation(X_batch, y_batch, weights, lambda_reg)
            cost = compute_cost(X_batch, y_batch, weights, lambda_reg)
            total_cost += cost

            # Update weights using gradients
            for l in range(len(weights)):
                weights[l] -= learning_rate * gradients[l]

        # Average cost over all mini-batches
        total_cost /= len(mini_batches)
        print(f"Iteration {iteration + 1}, Cost: {total_cost:.5f}")

        # Check stopping criterion
        if abs(prev_cost - total_cost) < epsilon:
            print(f"Stopping training as the improvement in cost is less than epsilon ({epsilon}).")
            break

        prev_cost = total_cost  # Update the previous cost

    # print("\n--------------------------------------------")
    # print("Running backpropagation")
    # print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    # for i, grad in enumerate(gradients):
    #     print(f"\t\tFinal regularized gradients of Theta {i + 1}:")
    #     for row in grad:
    #         print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))

    return weights, gradients

def normalize_data(X):
    """
    Normalize the input data to have zero mean and unit variance.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero for constant features
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def load_and_preprocess_dataset(file_path):
    """
    Load and preprocess the dataset from a CSV file.
    Assumes the last column contains the labels.
    """
    # Load the dataset
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

    # Split into features (X) and labels (y)
    X = data[:, :-1]  # All columns except the last
    y = data[:, -1]   # Last column

    # Normalize the features
    X_normalized, mean, std = normalize_data(X)

    # Convert labels to one-hot encoding (if binary classification, keep as is)
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    return X_normalized, y

if __name__ == "__main__":
    # Define the structure of the network
    layer_sizes = [30, 16, 8, 1]  # Example: 30 input features, 2 hidden layers, 1 output neuron
    learning_rate = 0.5
    max_iterations = 1000
    lambda_reg = 0.25
    batch_size = 32  # Mini-batch size
    epsilon = 1e-6  # Stopping criterion threshold

    # Load and preprocess the dataset
    file_path = "datasets\wdbc.csv"  # Replace with the actual path to your dataset
    X_train, y_train = load_and_preprocess_dataset(file_path)

    # Initialize weights
    weights = initialize_weights(layer_sizes)

    print(f"Regularization parameter lambda={lambda_reg:.3f}")
    print(f"\nInitializing the network with the following structure (number of neurons per layer): {layer_sizes}")

    # Train the network
    trained_weights, gradients = train_network(
        X_train, y_train,
        weights=weights,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        max_iterations=max_iterations,
        batch_size=batch_size,
        epsilon=epsilon
    )