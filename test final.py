import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

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
    """
    Adjust the dimensions of the weights for each layer.
    Initialize weights with small random numbers between -1 and +1.
    """
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
            # Initialize weights with small random numbers between -1 and +1
            adjusted_weights.append(np.random.uniform(-1, 1, size=(layer_sizes[i + 1], layer_sizes[i] + 1)))
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
            a = add_bias(a)  # Add bias term to activations
        a_values.append(a)

    return a_values, z_values

def compute_cost(X, y, weights, lambda_reg, verbose=False):
    m = X.shape[0]
    a_values, z_values = forward_propagation(X, weights)
    a_final = a_values[-1]
    a_final = np.clip(a_final, 1e-10, 1 - 1e-10)

    cost_per_example = -y * np.log(a_final) - (1 - y) * np.log(1 - a_final)
    total_cost = np.sum(cost_per_example) / m

    # Regularization term (exclude bias weights)
    reg_term = 0
    for theta in weights:
        reg_term += np.sum(theta[:, 1:]**2)
    reg_term = (lambda_reg / (2 * m)) * reg_term
    total_cost += reg_term

    return total_cost

def backward_propagation(X, y, weights, lambda_reg):
    m = X.shape[0]
    a_values, z_values = forward_propagation(X, weights)
    deltas = [a_values[-1] - y]  # Output layer delta

    # Backpropagate through hidden layers
    for l in range(len(weights) - 1, 0, -1):
        delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoid_derivative(a_values[l][:, 1:])
        deltas.append(delta)

    deltas.reverse()

    gradients = []
    for l in range(len(weights)):
        # Calculate gradient without regularization
        grad = np.dot(deltas[l].T, a_values[l]) / m
        
        # Add regularization (except for bias term)
        if lambda_reg != 0:
            reg_term = (lambda_reg / m) * weights[l]
            reg_term[:, 0] = 0  # Don't regularize bias term
            grad += reg_term
            
        gradients.append(grad)

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
    y = np.array(y).reshape(-1, y.shape[1]) 

    # print("\nInitial Weights:")
    # for i, theta in enumerate(weights):
    #     print(f"Theta {i + 1}:")
    #     for row in theta:
    #         print("\t" + "  ".join(f"{val:.5f}" for val in row))

    # print("\nTraining set:")
    # for i in range(len(X)):
    #     print(f"\tTraining instance {i + 1}")
    #     print(f"\tx: {X[i]}")
    #     print(f"\ty: {y[i]}")

    # # print("\n--------------------------------------------")
    # # print("Computing the error/cost, J, of the network")

    prev_cost = float('inf')  # Initialize the previous cost to infinity
    
    iteration = 0
    while True:
        iteration += 1
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

# def normalize_data(X):
#     """
#     Normalize the input data to have zero mean and unit variance.
#     """
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     std[std == 0] = 1  # Avoid division by zero for constant features
#     X_normalized = (X - mean) / std
#     return X_normalized, mean, std

def normalize_data(X):
    """
    Normalize the input data to scale values between [-1, +1] manually.
    """
    # Compute the min and max for each feature
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # Avoid division by zero for constant features
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Prevent division by zero

    # Scale the data to [-1, +1]
    X_normalized = 2 * (X - X_min) / X_range - 1

    return X_normalized, X_min, X_max

def load_and_preprocess_dataset(file_path):
    """
    Load and preprocess the dataset from a CSV file.
    Assumes the first row contains column names.
    If a column named 'label' exists, it is used as the label column.
    Otherwise, the last column is used as the label.
    """
    # Load the dataset using pandas
    data = pd.read_csv(file_path)

    # Define column names
    column_names = data.columns.tolist()

    # Check if a 'label' column exists
    if 'label' in column_names:
        label_column = 'label'
    else:
        label_column = column_names[-1]  # Default to the last column

    # Split into features (X) and labels (y)
    X = data.drop(columns=[label_column]).copy()  # All columns except the label
    y = data[label_column].values  # Label column

    # Identify categorical and numerical columns
    categorical_columns = [col for col in X.columns if '_cat' in col]
    numerical_columns = [col for col in X.columns if '_num' in col]

    # One-hot encode categorical columns
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical_encoded = encoder.fit_transform(X[categorical_columns])
        X_categorical_encoded = pd.DataFrame(X_categorical_encoded, index=X.index)
    else:
        X_categorical_encoded = pd.DataFrame(index=X.index)  # Empty DataFrame if no categorical columns

    # Keep numerical columns as is
    X_numerical = X[numerical_columns]

    # Combine numerical and one-hot encoded categorical columns
    X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1).values

    # Normalize the numerical features
    X_normalized, mean, std = normalize_data(X_processed)

    # Convert labels to a column vector
    y = y.reshape(-1, 1)

    return X_normalized, y

def calculate_metrics(y_true, y_pred):
    """
    Calculate Accuracy, Precision, Recall, and F1 Score from scratch.
    """
    # Convert predictions to binary (threshold = 0.5)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def predict(X, weights):
    """
    Make predictions using the trained neural network.
    """
    a_values, _ = forward_propagation(X, weights)
    y_pred = a_values[-1]  # Final layer activations
    return y_pred

def stratified_k_fold_split(X, y, k=5):
    """
    Perform stratified k-fold splitting of the dataset without using libraries.
    Ensures that each fold has a similar class distribution as the original dataset.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        k (int): Number of folds.

    Returns:
        List of tuples: Each tuple contains (X_train, y_train, X_test, y_test) for a fold.
    """
    # Get unique classes and their indices
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    # Shuffle indices for each class
    for cls in classes:
        np.random.shuffle(class_indices[cls])

    # Split indices for each class into k folds
    folds = [[] for _ in range(k)]
    for cls in classes:
        cls_indices = class_indices[cls]
        cls_folds = np.array_split(cls_indices, k)
        for i in range(k):
            folds[i].extend(cls_folds[i])

    # Create train-test splits for each fold
    splits = []
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        splits.append((X_train, y_train, X_test, y_test))

    return splits

def process_network_case(case):
    """
    Process the network for the given case and print the required outputs.

    Parameters:
        case (int): The case to process (1 for the first case, 2 for the second case).

    Prints:
        - Activation of each neuron
        - Final predicted output of the network
        - Expected output
        - Cost (J) associated with each instance
        - Delta values of each neuron
        - Gradients of all weights after processing each instance
        - Final (regularized) gradients after backpropagation
        - Final (regularized) cost, J, based on the complete training set
    """
    if case == 1:
        # Case 1: Network Structure [1, 2, 1]
        network_structure = [1, 2, 1]
        weights_list = [
            np.array([[0.40000, 0.10000], [0.30000, 0.20000]]),  # Theta1
            np.array([[0.70000, 0.50000, 0.60000]])          # Theta2
        ]
        training_data = [
            (np.array([0.13000]), np.array([0.90000])),  # Training instance 1
            (np.array([0.42000]), np.array([0.23000]))  # Training instance 2
        ]
        lambda_reg = 0.0
    elif case == 2:
        # Case 2: Network Structure [2, 4, 3, 2]
        network_structure = [2, 4, 3, 2]
        weights_list = [
            np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]),  # Theta1
            np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]),  # Theta2
            np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])  # Theta3
        ]
        training_data = [
            (np.array([0.32, 0.68]), np.array([0.75, 0.98])),  # Training instance 1
            (np.array([0.83, 0.02]), np.array([0.75, 0.28]))   # Training instance 2
        ]
        lambda_reg = 0.25
    else:
        raise ValueError("Invalid case. Please pass 1 for the first case or 2 for the second case.")

    # Process each training instance
    for instance_idx, (x, y) in enumerate(training_data):
        print(f"\nProcessing Training Instance {instance_idx + 1}")
        print(f"Input x: {x}")
        print(f"Expected Output y: {y}")

        # Forward propagation
        a_values, z_values = forward_propagation(x, weights_list)
        y_pred = a_values[-1]
        cost = compute_cost(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_reg=lambda_reg)

        print("\nActivations of Each Neuron:")
        for layer_idx, a in enumerate(a_values):
            print(f"Layer {layer_idx + 1} Activations: {a}")

        print(f"\nFinal Predicted Output: {y_pred}")
        print(f"Cost (J) for this instance: {cost:.5f}")

        # Backward propagation
        gradients = backward_propagation(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_reg=lambda_reg)

        print("\nDelta Values of Each Neuron:")
        for layer_idx, z in enumerate(z_values):
            delta = sigmoid_derivative(z)
            print(f"Layer {layer_idx + 1} Delta Values: {delta}")

        print("\nGradients of All Weights After Processing This Instance:")
        for layer_idx, grad in enumerate(gradients):
            print(f"Gradient for Theta{layer_idx + 1}:")
            print(grad)

    # Final regularized gradients after processing all instances
    final_gradients = backward_propagation(
        np.array([x for x, _ in training_data]),
        np.array([y for _, y in training_data]),
        weights_list,
        lambda_reg=lambda_reg
    )

    print("\nFinal (Regularized) Gradients After Backpropagation:")
    for layer_idx, grad in enumerate(final_gradients):
        print(f"Regularized Gradient for Theta{layer_idx + 1}:")
        print(grad)

    # Compute the final (regularized) cost based on the complete training set
    X_train = np.array([x for x, _ in training_data])
    y_train = np.array([y for _, y in training_data])
    final_cost = compute_cost(X_train, y_train, weights_list, lambda_reg=lambda_reg)
    
    print(f"\nFinal (Regularized) Cost, J, Based on the Complete Training Set: {final_cost:.5f}")

    # print("\nGradients of All Weights After Processing This Instance:")
    # for layer_idx, grad in enumerate(gradients):
    #     print(f"Gradient for Theta{layer_idx + 1}:")
    # # Round each value to 5 decimal places when printing
    # for row in grad:
    #     print("\t" + "  ".join(f"{val:.5f}" for val in row))

    # print("\nFinal (Regularized) Gradients After Backpropagation:")
    # for layer_idx, grad in enumerate(final_gradients):
    #     print(f"Regularized Gradient for Theta{layer_idx + 1}:")
        
    # # Round each value to 5 decimal places when printing
    # for row in grad:
    #     print("\t" + "  ".join(f"{val:.5f}" for val in row))

    # print(f"\nFinal (Regularized) Cost, J, Based on the Complete Training Set: {final_cost:.5f}")

# if __name__ == "__main__":
#     # Load and preprocess the dataset
#     file_path = "wdbc.csv"  # Replace with the actual path to your dataset
#     X, y = load_and_preprocess_dataset(file_path)

#     input_layer_size = X.shape[1]

#     # Define the structure of the network
#     layer_sizes = [input_layer_size, 16, 8, 1]
#     learning_rate = 0.5
#     max_iterations = 1000
#     lambda_reg = 0.25
#     batch_size = 32  # Mini-batch size
#     epsilon = 1e-6  # Stopping criterion threshold
#     k = 5  # Number of folds for cross-validation

#     # Perform stratified k-fold cross-validation
#     splits = stratified_k_fold_split(X, y, k=k)

#     # Initialize metrics
#     fold_metrics = []  # To store metrics for each fold
#     total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0

#     for fold, (train_indices, test_indices) in enumerate(splits):
#         print(f"\nProcessing Fold {fold + 1}/{k}")

#         # Split the data into training and testing sets
#         X_train, y_train = X[train_indices], y[train_indices]
#         X_test, y_test = X[test_indices], y[test_indices]

#         # Initialize weights
#         weights = initialize_weights(layer_sizes)

#         # Train the network
#         trained_weights, gradients = train_network(
#             X_train, y_train,
#             weights=weights,
#             learning_rate=learning_rate,
#             lambda_reg=lambda_reg,
#             max_iterations=max_iterations,
#             batch_size=batch_size,
#             epsilon=epsilon
#         )

#         # Make predictions on the test set
#         y_pred = predict(X_test, trained_weights)

#         # Calculate metrics for this fold
#         accuracy, precision, recall, f1_score = calculate_metrics(y_test, y_pred)

#         # Store metrics for this fold
#         fold_metrics.append({
#             "Fold": fold + 1,
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1_score
#         })

#         # Accumulate metrics
#         total_accuracy += accuracy
#         total_precision += precision
#         total_recall += recall
#         total_f1_score += f1_score

#     # Compute average metrics across all folds
#     avg_accuracy = total_accuracy / k
#     avg_precision = total_precision / k
#     avg_recall = total_recall / k
#     avg_f1_score = total_f1_score / k

#     # Print metrics for each fold
#     print("\nMetrics for Each Fold:")
#     for metrics in fold_metrics:
#         print(f"Fold {metrics['Fold']} Metrics:")
#         print(f"Accuracy: {metrics['Accuracy']:.4f}")
#         print(f"Precision: {metrics['Precision']:.4f}")
#         print(f"Recall: {metrics['Recall']:.4f}")
#         print(f"F1 Score: {metrics['F1 Score']:.4f}")
#         print(f"\n")

#     # Print average metrics
#     print("\nAverage Metrics Across All Folds:")
#     print(f"Accuracy: {avg_accuracy:.4f}")
#     print(f"Precision: {avg_precision:.4f}")
#     print(f"Recall: {avg_recall:.4f}")
#     print(f"F1 Score: {avg_f1_score:.4f}")


if __name__ == "__main__":
    # Variable to call the function for specific cases
    case = None  # Set to 1 for the first case, 2 for the second case, or None to do nothing
    mode = 2
    
    # Load and preprocess the dataset
    file_path = "wdbc.csv"  # Replace with the actual path to your dataset
    X, y = load_and_preprocess_dataset(file_path)

    input_layer_size = X.shape[1]

    # Define the structure of the network
    layer_sizes = [input_layer_size, 1]
    learning_rate = 0.5
    # max_iterations = 1000
    lambda_reg = 0.25
    batch_size = 32  # Mini-batch size
    epsilon = 0.0001  # Stopping criterion threshold
    k = 5  # Number of folds for cross-validation

    if case is not None:
        process_network_case(case)
    elif mode == 1:
        # # Load and preprocess the dataset
        # file_path = "datasets/wdbc.csv"  # Replace with the actual path to your dataset
        # X, y = load_and_preprocess_dataset(file_path)

        # input_layer_size = X.shape[1]

        # # Define the structure of the network
        # layer_sizes = [input_layer_size, 1]
        # learning_rate = 0.5
        # # max_iterations = 1000
        # lambda_reg = 0.25
        # batch_size = 32  # Mini-batch size
        # epsilon = 0.0001  # Stopping criterion threshold
        # k = 5  # Number of folds for cross-validation

        # Perform stratified k-fold cross-validation
        splits = stratified_k_fold_split(X, y, k=k)

        # Initialize metrics
        fold_metrics = []  # To store metrics for each fold
        total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0

        for fold, (X_train, y_train, X_test, y_test) in enumerate(splits):
            print(f"\nProcessing Fold {fold + 1}/{k}")

            # Initialize weights
            weights = initialize_weights(layer_sizes)

            # Train the network
            trained_weights, gradients = train_network(
                X_train, y_train,
                weights=weights,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                batch_size=batch_size,
                epsilon=epsilon
            )

            # Make predictions on the test set
            y_pred = predict(X_test, trained_weights)

            # Calculate metrics for this fold
            accuracy, precision, recall, f1_score = calculate_metrics(y_test, y_pred)

            # Store metrics for this fold
            fold_metrics.append({
                "Fold": fold + 1,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            })

            # Accumulate metrics
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

        # Compute average metrics across all folds
        avg_accuracy = total_accuracy / k
        avg_precision = total_precision / k
        avg_recall = total_recall / k
        avg_f1_score = total_f1_score / k

        # Print metrics for each fold
        print("\nMetrics for Each Fold:")
        for metrics in fold_metrics:
            print(f"Fold {metrics['Fold']} Metrics:")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1 Score: {metrics['F1 Score']:.4f}")
            print(f"\n")

        # Print average metrics
        print("\nAverage Metrics Across All Folds:")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1 Score: {avg_f1_score:.4f}")
    elif mode == 2:
        # Load and preprocess the dataset
        # file_path = "datasets/wdbc.csv"  # Replace with the actual path to your dataset
        # X, y = load_and_preprocess_dataset(file_path)

        # input_layer_size = X.shape[1]

        # # Define the structure of the network
        # layer_sizes = [input_layer_size, 16, 8, 1]
        # learning_rate = 0.5  # Step size (α)
        # #max_iterations = 50
        # lambda_reg = 0.25
        # batch_size = 32  # Mini-batch size
        # epsilon = 1e-6  # Stopping criterion threshold

        # Split the dataset into training and test sets
        split_ratio = 0.8  # 80% training, 20% testing
        split_index = int(split_ratio * X.shape[0])
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Initialize variables for the learning curve
        training_sizes = []
        test_costs = []

        # Incrementally increase the number of training samples
        for num_samples in range(5, X_train.shape[0] + 1, 5):  # Increase by 10 samples at a time
            print(f"\nTraining with {num_samples} samples...")

            # Use the first 'num_samples' training examples
            X_train_subset = X_train[:num_samples]
            y_train_subset = y_train[:num_samples]

            # Initialize weights
            weights = initialize_weights(layer_sizes)

            # Train the network
            trained_weights, gradients = train_network(
                X_train_subset, y_train_subset,
                weights=weights,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                # max_iterations=max_iterations,
                batch_size=batch_size,
                epsilon=epsilon
            )

            # Compute the cost on the test set
            test_cost = compute_cost(X_test, y_test, trained_weights, lambda_reg)
            print(f"Test Cost (J) with {num_samples} training samples: {test_cost:.5f}")

            # Store the results
            training_sizes.append(num_samples)
            test_costs.append(test_cost)

        # Plot the learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, test_costs, marker='o', label="Test Cost (J)")
        plt.title("Learning Curve")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Cost Function (J)")
        plt.grid(True)
        plt.legend()
        plt.show()