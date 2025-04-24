import numpy as np
import pandas as pd
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
        reg_term[:, 0] = 0  # Exclude bias weights from regularization
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
    Assumes the first row contains column names and the last column contains the labels.
    """
    # Load the dataset using pandas
    data = pd.read_csv(file_path)

    # Define column names
    column_names = data.columns.tolist()

    # Split into features (X) and labels (y)
    X = data.iloc[:, :-1].copy()  # All columns except the last (label)
    y = data.iloc[:, -1].values   # Last column (label)

    # Identify categorical and numerical columns
    categorical_columns = [col for col in column_names[:-1] if '_cat' in col]
    numerical_columns = [col for col in column_names[:-1] if '_num' in col]

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
    Perform stratified k-fold splitting of the dataset.
    Ensures that each fold has a similar class distribution as the original dataset.
    """
    # Get the unique classes and their indices
    classes, y_indices = np.unique(y, return_inverse=True)
    folds = [[] for _ in range(k)]

    # Split indices for each class into k folds
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        cls_folds = np.array_split(cls_indices, k)
        for i in range(k):
            folds[i].extend(cls_folds[i])

    # Create train-test splits for each fold
    splits = []
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        splits.append((train_indices, test_indices))

    return splits

if __name__ == "__main__":
    # Load and preprocess the dataset
    file_path = "wdbc.csv"  # Replace with the actual path to your dataset
    X, y = load_and_preprocess_dataset(file_path)

    input_layer_size = X.shape[1]

    # Define the structure of the network
    layer_sizes = [input_layer_size, 16, 8, 1]
    learning_rate = 0.5
    max_iterations = 1000
    lambda_reg = 0.25
    batch_size = 32  # Mini-batch size
    epsilon = 1e-6  # Stopping criterion threshold
    k = 5  # Number of folds for cross-validation

    # Perform stratified k-fold cross-validation
    splits = stratified_k_fold_split(X, y, k=k)

    # Initialize metrics
    fold_metrics = []  # To store metrics for each fold
    total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0

    for fold, (train_indices, test_indices) in enumerate(splits):
        print(f"\nProcessing Fold {fold + 1}/{k}")

        # Split the data into training and testing sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Initialize weights
        weights = initialize_weights(layer_sizes)

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