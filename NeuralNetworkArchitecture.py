import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(a):
    return a * (1 - a)

def addBiasNeuron(X):
    if X.ndim == 1:
        return np.concatenate([[1], X])
    return np.column_stack([np.ones(X.shape[0]), X])

def generateInitialWeights(layer_sizes):
    adjusted_weights = []
    for i in range(len(layer_sizes) - 1):
        adjusted_weights.append(np.random.uniform(-1, 1, size=(layer_sizes[i + 1], layer_sizes[i] + 1))) # Initialize weights (-1 and +1)
    return adjusted_weights

def forwardPropagation(X, weights):
    a_values = [addBiasNeuron(X)] # Add bias to input layer
    z_values = []

    for i, theta in enumerate(weights):
        z = np.dot(a_values[-1], theta.T)
        z_values.append(z)
        a = sigmoid(z)

        if i < len(weights) - 1: # Only add bias if not the last layer
            a = addBiasNeuron(a) # Add bias term to activations
        a_values.append(a)

    return a_values, z_values

def calculateCost(X, y, weights, lambda_reg, verbose=False):
    m = X.shape[0]
    a_values, _ = forwardPropagation(X, weights)
    a_final = a_values[-1]
    a_final = np.clip(a_final, 1e-10, 1 - 1e-10)

    cost_per_example = -y * np.log(a_final) - (1 - y) * np.log(1 - a_final)
    total_cost = np.sum(cost_per_example) / m

    reg_term = 0 # Regularization term (exclude bias weights)
    for theta in weights:
        reg_term += np.sum(theta[:, 1:]**2)
    reg_term = (lambda_reg / (2 * m)) * reg_term
    total_cost += reg_term

    return total_cost

def backwardPropagation(X, y, weights, lambda_reg):
    m = X.shape[0]
    a_values, _ = forwardPropagation(X, weights)
    deltas = [a_values[-1] - y] # Output layer delta

    for l in range(len(weights) - 1, 0, -1):
        delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoidDerivative(a_values[l][:, 1:])
        deltas.append(delta)

    deltas.reverse()

    gradients = []
    for l in range(len(weights)): # Calculate gradient without regularization
        grad = np.dot(deltas[l].T, a_values[l]) / m
        
        if lambda_reg != 0: # Add regularization (except for bias term)
            reg_term = (lambda_reg / m) * weights[l]
            reg_term[:, 0] = 0 # Don't regularize bias term
            grad += reg_term
            
        gradients.append(grad)

    return gradients

def generateMiniBatches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m) # Shuffle the data
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

def trainModel(X, y, weights, alpha_learning_rate=0.5, lambda_reg=0.0, max_iterations=1, batch_size=1, epsilon=1e-6):
    y = np.array(y).reshape(-1, y.shape[1]) 

    # prev_cost = float('inf') # Initialize the intial cost to infinity
    
    for iteration in range(max_iterations):
        mini_batches = generateMiniBatches(X, y, batch_size) # Create mini-batches
        total_cost = 0

        for X_batch, y_batch in mini_batches:
            gradients = backwardPropagation(X_batch, y_batch, weights, lambda_reg)
            cost = calculateCost(X_batch, y_batch, weights, lambda_reg)
            total_cost += cost

            for l in range(len(weights)):
                weights[l] -= alpha_learning_rate * gradients[l]

        total_cost /= len(mini_batches)
        print(f"Iteration: {iteration + 1}, Cost: {total_cost:.5f}")

        # if abs(prev_cost - total_cost) < epsilon:
        #     print(f"Training Stopped - Less Improvement in Cost than {epsilon}.")
        #     break

        # prev_cost = total_cost # Update the previous cost

    return weights, gradients

def dataNormalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1 # Prevent division by zero
    
    X_normalized = 2 * (X - X_min) / X_range - 1 # Scale the data to [-1, +1]

    return X_normalized, X_min, X_max

def preprocessDataset(file_path):
    data = pd.read_csv(file_path)

    column_names = data.columns.tolist()

    if 'label' in column_names:
        label_column = 'label'
    else:
        label_column = column_names[-1] # Default to the last column

    X = data.drop(columns=[label_column]).copy() # All columns except the label
    y = data[label_column].values # Label column

    categorical_columns = [col for col in X.columns if '_cat' in col]
    numerical_columns = [col for col in X.columns if '_num' in col]

    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical_encoded = encoder.fit_transform(X[categorical_columns])
        X_categorical_encoded = pd.DataFrame(X_categorical_encoded, index=X.index)
    else:
        X_categorical_encoded = pd.DataFrame(index=X.index) # Empty DataFrame if no categorical columns

    X_numerical = X[numerical_columns]
    X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1).values
    X_normalized, mean, std = dataNormalization(X_processed)

    y = y.reshape(-1, 1)

    return X_normalized, y

def calculateEvaluationMetrics(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)

    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def predictClass(X, weights):
    a_values, _ = forwardPropagation(X, weights)
    y_pred = a_values[-1] # Final layer activations
    return y_pred

def stratifiedKFoldCrossValidation(X, y, k=5):
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    for cls in classes:
        np.random.shuffle(class_indices[cls])

    folds = [[] for _ in range(k)]
    for cls in classes:
        cls_indices = class_indices[cls]
        cls_folds = np.array_split(cls_indices, k)
        for i in range(k):
            folds[i].extend(cls_folds[i])

    splits = []
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        splits.append((X_train, y_train, X_test, y_test))

    return splits

def testCase(case):
    if case == 1: # Case 1: Network Structure [1, 2, 1]
        weights_list = [
            np.array([[0.40000, 0.10000], [0.30000, 0.20000]]),  # Theta1
            np.array([[0.70000, 0.50000, 0.60000]])              # Theta2
        ]
        training_data = [
            (np.array([0.13000]), np.array([0.90000])),  # Training instance 1
            (np.array([0.42000]), np.array([0.23000]))   # Training instance 2
        ]
        lambda_reg = 0.0

    elif case == 2: # Case 2: Network Structure [2, 4, 3, 2]
        weights_list = [
            np.array([[0.42000, 0.15000, 0.40000], [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000], [0.30000, 0.35000, 0.68000]]),                           # Theta1
            np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000], [0.87000, 0.42000, 0.20000, 0.32000, 0.89000], [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]]),  # Theta2
            np.array([[0.04000, 0.87000, 0.42000, 0.53000], [0.17000, 0.10000, 0.95000, 0.69000]])                                                                    # Theta3
        ]
        training_data = [
            (np.array([0.32000, 0.68000]), np.array([0.75000, 0.98000])),  # Training instance 1
            (np.array([0.83000, 0.02000]), np.array([0.75000, 0.28000]))   # Training instance 2
        ]
        lambda_reg = 0.25
    else:
        raise ValueError("Invalid case. Please press 1 for the first case or 2 for the second case.")

    for instance_idx, (x, y) in enumerate(training_data):
        print(f"\nProcessing Training Instance {instance_idx + 1}")
        print(f"Input x: {[f'{val:.5f}' for val in x]}")
        print(f"Expected Output y: {[f'{val:.5f}' for val in y]}")

        a_values, z_values = forwardPropagation(x, weights_list)
        y_pred = a_values[-1]
        cost = calculateCost(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_reg=lambda_reg)

        print("\nActivations of Each Neuron:")
        for layer_idx, a in enumerate(a_values):
            print(f"Layer {layer_idx + 1} Activations: {[f'{val:.5f}' for val in a]}")

        print(f"\nFinal Predicted Output: {[f'{val:.5f}' for val in y_pred]}")
        print(f"Cost (J) for this instance: {cost:.5f}")

        gradients = backwardPropagation(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_reg=lambda_reg)

        print("\nDelta Values of Each Neuron:")
        for layer_idx, z in enumerate(z_values):
            delta = sigmoidDerivative(z)
            print(f"Layer {layer_idx + 1} Delta Values: {[f'{val:.5f}' for val in delta]}")

        print("\nGradients of All Weights After Processing This Instance:")
        for layer_idx, grad in enumerate(gradients):
            print(f"Gradient for Theta{layer_idx + 1}:")
            for row in grad:
                print("\t" + "  ".join(f"{val:.5f}" for val in row))

    final_gradients = backwardPropagation(
        np.array([x for x, _ in training_data]),
        np.array([y for _, y in training_data]),
        weights_list,
        lambda_reg=lambda_reg
    )

    print("\nFinal (Regularized) Gradients After Backpropagation:")
    for layer_idx, grad in enumerate(final_gradients):
        print(f"Regularized Gradient for Theta{layer_idx + 1}:")
        for row in grad:
            print("\t" + "  ".join(f"{val:.5f}" for val in row))

    X_train = np.array([x for x, _ in training_data])
    y_train = np.array([y for _, y in training_data])
    final_cost = calculateCost(X_train, y_train, weights_list, lambda_reg=lambda_reg)
    
    print(f"\nFinal (Regularized) Cost, J, Based on the Complete Training Set: {final_cost:.5f}")

if __name__ == "__main__":
    case = None # Set to 1 for the first case, 2 for the second case, or None to do nothing
    mode = 2

    file_path = "wdbc.csv" # Change this for using different datasets
    X, y = preprocessDataset(file_path)

    indices = np.random.permutation(X.shape[0]) 
    X = X[indices] 
    y = y[indices] 

    layer_sizes = [X.shape[1], 16, 1]
    alpha_learning_rate=0.1
    lambda_reg=0.01
    max_iterations=100
    batch_size=32
    # epsilon=0.00001
    k=5

    if case is not None:
        testCase(case)
    elif mode == 1:
        splits = stratifiedKFoldCrossValidation(X, y, k=k)

        fold_metrics = [] # To store metrics for each fold
        total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0


        for fold, (X_train, y_train, X_test, y_test) in enumerate(splits):
            print(f"\nProcessing Fold {fold + 1}/{k}")

            weights = generateInitialWeights(layer_sizes)

            trained_weights, gradients = trainModel(
                X_train, y_train,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_reg=lambda_reg,
                max_iterations=max_iterations,
                batch_size=batch_size,
                # epsilon=epsilon
            )

            y_pred = predictClass(X_test, trained_weights)

            accuracy, precision, recall, f1_score = calculateEvaluationMetrics(y_test, y_pred)

            fold_metrics.append({
                "Fold": fold + 1,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            })

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

        avg_accuracy = total_accuracy / k
        avg_precision = total_precision / k
        avg_recall = total_recall / k
        avg_f1_score = total_f1_score / k

        print("\nMetrics for Each Fold:")
        for metrics in fold_metrics:
            print(f"Fold {metrics['Fold']} Metrics:")
            print(f"Accuracy: {metrics['Accuracy'] * 100:.5f}")
            print(f"Precision: {metrics['Precision'] * 100:.5f}")
            print(f"Recall: {metrics['Recall'] * 100:.5f}")
            print(f"F1 Score: {metrics['F1 Score'] * 100:.5f}")
            print(f"\n")

        print("\nAverage Metrics Across All Folds:")
        print(f"Accuracy: {avg_accuracy * 100:.5f}")
        print(f"Precision: {avg_precision * 100:.5f}")
        print(f"Recall: {avg_recall * 100:.5f}")
        print(f"F1 Score: {avg_f1_score * 100:.5f}")

    elif mode == 2:
        split_ratio = 0.8 # 80% training, 20% testing
        split_index = int(split_ratio * X.shape[0])
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        training_sizes = []
        test_costs = []

        weights = generateInitialWeights(layer_sizes)

        for num_samples in range(10, X_train.shape[0] + 1, 10): # Increase by 10 samples at a time
            print(f"\nTraining with {num_samples} samples.")

            X_train_subset = X_train[:num_samples]
            y_train_subset = y_train[:num_samples]

            # weights = generateInitialWeights(layer_sizes)

            trained_weights, gradients = trainModel(
                X_train_subset, y_train_subset,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_reg=lambda_reg,
                max_iterations=max_iterations,
                batch_size=batch_size,
                # epsilon=epsilon
            )

            test_cost = calculateCost(X_test, y_test, trained_weights, lambda_reg)
            print(f"Test Cost (J) with {num_samples} training samples: {test_cost:.5f}")

            training_sizes.append(num_samples)
            test_costs.append(test_cost)
        
        dataset_title = (file_path.split("/")[-1].split(".")[0]).capitalize() # Extract dataset title from the file path

        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, test_costs, label="Cost (J)", linewidth=2)
        plt.title(f"Learning Curve - NN: {layer_sizes} - {dataset_title} Dataset - Lambda: {lambda_reg} - Alpha: {alpha_learning_rate} - Batch Size: {batch_size}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Cost Function (J)")
        # plt.grid(True)
        plt.legend()
        plt.show()