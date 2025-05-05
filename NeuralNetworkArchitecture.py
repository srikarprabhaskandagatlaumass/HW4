# Srikar Prabhas Kandagatla (34964700)
"""
This code implements the Neural Network in Python to classify WDBC, Loan,
Titanic, and Raisins dataset.

I executed the code using the VS Code IDE. To run it on your local machine, download the
dataset (four datasets), set the working directory, and execute the script.

This code was completed and submitted as part of Homework Assignment 4 for
COMPSCI 589: Machine Learning at the University of Massachusetts, Amherst for Spring'25 Semester.
"""

# COMPSCI 589: Machine Learning (Spring 2025)
"""
In this code, I have implemented the Neural Network Architecture to classify the WDBC, Loan,
Titanic, and Raisins datasets. This model also supports the Numeric and Categorical features which
can support varaible architectures.
"""

# Importing the Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Helper Functions
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
        weight_matrix = np.random.uniform(low=-1, high=1, size=(layer_sizes[i + 1], layer_sizes[i] + 1))
        adjusted_weights.append(weight_matrix)  
    return adjusted_weights

# Forward and Backward Propagation Functions to calculate the cost and gradients_regularizationients in the Neural Network done trainModel function
def forwardPropagation(X, weights):
    a_values = [addBiasNeuron(X)] 
    z_values = []

    for i, theta in enumerate(weights):
        z = np.dot(a_values[-1], theta.T)
        z_values.append(z)
        a = sigmoid(z)

        if i < len(weights) - 1: 
            a = addBiasNeuron(a) 
        a_values.append(a)

    return a_values, z_values

def calculateCost(X, y, weights, lambda_regulation):
    m = X.shape[0]  
    a_values, _ = forwardPropagation(X, weights)
    a_final = a_values[-1]
    a_final = np.clip(a_final, 1e-10, 1 - 1e-10)

    cost_per_example = y * (-np.log(a_final)) + (1 - y) * (-np.log(1 - a_final))
    total_cost = np.sum(cost_per_example) / m  

    if m > 1: 
        regularization = 0
        for theta in weights:
            regularization += np.sum(theta[:, 1:]**2)  
        regularization = (lambda_regulation / (2 * m)) * regularization
        total_cost += regularization

    return total_cost

def backwardPropagation(X, y, weights, lambda_regulation):
    m = X.shape[0] 
    a_values, _ = forwardPropagation(X, weights)
    deltas = [a_values[-1] - y]  

    for l in range(len(weights) - 1, 0, -1):
        if a_values[l].ndim == 1: 
            delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoidDerivative(a_values[l][1:])
        else: 
            delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoidDerivative(a_values[l][:, 1:])
        deltas.append(delta)

    deltas.reverse()

    gradients_regularizationients = []
    for l in range(len(weights)):
        gradients_regularization = np.dot(deltas[l].T, a_values[l]) / m
        
        if lambda_regulation != 0:
            regularization = (lambda_regulation / m) * weights[l]
            regularization[:, 0] = 0 
            gradients_regularization += regularization
        
        gradients_regularizationients.append(gradients_regularization)

    return gradients_regularizationients

def generateMiniBatches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m) 
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

# Compleented the epsilon stopping creiteria and implemented the maximum iterations, no commeent and change the code loop to while loop to use epsilion stopping criteria
def trainModel(X, y, weights, alpha_learning_rate=0.5, lambda_regulation=0.0, max_iterations=1, batch_size=1, epsilon=1e-6):
    y = np.array(y).reshape(-1, y.shape[1])
    
    for iteration in range(max_iterations):
        mini_batches = generateMiniBatches(X, y, batch_size)
        total_cost = 0
        accumulated_gradients = [np.zeros_like(w) for w in weights] 
        
        for X_batch, y_batch in mini_batches:
            gradients = backwardPropagation(X_batch, y_batch, weights, 0) 
            cost = calculateCost(X_batch, y_batch, weights, lambda_regulation)
            total_cost += cost
            
            for l in range(len(weights)):
                accumulated_gradients[l] += gradients[l] * len(X_batch)

        m = X.shape[0]
        for l in range(len(accumulated_gradients)):
            avg_gradients = accumulated_gradients[l] / m
            avg_gradients += (lambda_regulation / m) * weights[l]
            avg_gradients[:, 0] = accumulated_gradients[l][:, 0] / m 
            
            weights[l] -= alpha_learning_rate * avg_gradients
        
        total_cost /= len(mini_batches)
        print(f"Iteration: {iteration+1}, Cost: {total_cost:.5f}")
    
    return weights, gradients


def dataNormalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1 
    
    X_normalized = 2 * (X - X_min) / X_range - 1 # [-1, +1]

    return X_normalized, X_min, X_max

def preprocessDataset(dataset_path):
    data = pd.read_csv(dataset_path)

    column_names = data.columns.tolist()

    if 'label' in column_names:
        label_column = 'label'
    else:
        label_column = column_names[-1] # Default to the last column

    X = data.drop(columns=[label_column]).copy()
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
    X_normalized, _, _ = dataNormalization(X_processed)

    y = y.reshape(-1, 1)

    return X_normalized, y

def calculateEvaluationMetrics(y_true, y_pred):
    y_pred_limit = (y_pred >= 0.5).astype(int)

    TP = np.sum((y_true == 1) & (y_pred_limit == 1))
    FP = np.sum((y_true == 0) & (y_pred_limit == 1))
    TN = np.sum((y_true == 0) & (y_pred_limit == 0))
    FN = np.sum((y_true == 1) & (y_pred_limit == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def predictClass(X, weights):
    a_values, _ = forwardPropagation(X, weights)
    y_pred = a_values[-1] 
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
    if case == 1:  # Case 1: Network Structure [1, 2, 1]
        weights_list = [
            np.array([[0.40000, 0.10000], [0.30000, 0.20000]]),  # Theta1
            np.array([[0.70000, 0.50000, 0.60000]])              # Theta2
        ]
        training_data = [
            (np.array([0.13000]), np.array([0.90000])),  # Training instance 1
            (np.array([0.42000]), np.array([0.23000]))   # Training instance 2
        ]
        lambda_regulation = 0.0

    elif case == 2:  # Case 2: Network Structure [2, 4, 3, 2]
        weights_list = [
            np.array([[0.42000, 0.15000, 0.40000], [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000], [0.30000, 0.35000, 0.68000]]),                           # Theta1
            np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000], [0.87000, 0.42000, 0.20000, 0.32000, 0.89000], [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]]),  # Theta2
            np.array([[0.04000, 0.87000, 0.42000, 0.53000], [0.17000, 0.10000, 0.95000, 0.69000]])                                                                    # Theta3
        ]
        training_data = [
            (np.array([0.32000, 0.68000]), np.array([0.75000, 0.98000])),  # Training instance 1
            (np.array([0.83000, 0.02000]), np.array([0.75000, 0.28000]))   # Training instance 2
        ]
        lambda_regulation = 0.25

    print("--------------------------------------------")
    print("Computing the error/cost, J, of the network")

    for instance_idx, (x, y) in enumerate(training_data):
        print(f"\tProcessing training instance {instance_idx + 1}")
        print(f"\tForward propagating the input {[f'{val:.5f}' for val in x]}")

        a_values, z_values = forwardPropagation(x, weights_list)
        y_pred = a_values[-1]
        cost = calculateCost(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_regulation=lambda_regulation)

        for layer_idx, (a, z) in enumerate(zip(a_values, z_values + [None])):
            if z is not None:
                print(f"\t\tz{layer_idx + 2}: {[f'{val:.5f}' for val in z]}")
            print(f"\t\ta{layer_idx + 1}: {[f'{val:.5f}' for val in a]}\n")

        print(f"\n\t\tf(x): {[f'{val:.5f}' for val in y_pred]}")
        print(f"\tPredicted output for instance {instance_idx + 1}: {[f'{val:.5f}' for val in y_pred]}")
        print(f"\tExpected output for instance {instance_idx + 1}: {[f'{val:.5f}' for val in y]}")
        print(f"\tCost, J, associated with instance {instance_idx + 1}: {cost:.3f}\n")

    X_train = np.array([x for x, _ in training_data])
    y_train = np.array([y for _, y in training_data])
    final_cost = calculateCost(X_train, y_train, weights_list, lambda_regulation=lambda_regulation)
    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")
    print("--------------------------------------------")

    print("Running backpropagation")
    for instance_idx, (x, y) in enumerate(training_data):
        print(f"\tComputing gradients based on training instance {instance_idx + 1}")

        gradients = backwardPropagation(x.reshape(1, -1), y.reshape(1, -1), weights_list, lambda_regulation=lambda_regulation)
        deltas = [a_values[-1] - y] 
        for l in range(len(weights_list) - 1, 0, -1):
            if a_values[l].ndim == 1: 
                delta = np.dot(deltas[-1], weights_list[l][:, 1:]) * sigmoidDerivative(a_values[l][1:])
            else: 
                delta = np.dot(deltas[-1], weights_list[l][:, 1:]) * sigmoidDerivative(a_values[l][:, 1:])
            deltas.append(delta)

        deltas.reverse()

        for layer_idx, delta in enumerate(deltas):
            print(f"\t\tdelta{layer_idx + 2}: {[f'{val:.5f}' for val in delta]}") 

        for layer_idx, gradient in enumerate(gradients):
            print(f"\n\t\tGradients of Theta{layer_idx + 1} based on training instance {instance_idx + 1}:")
            for row in gradient:
                print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))

    final_gradients = backwardPropagation(X_train, y_train, weights_list, lambda_regulation=lambda_regulation)
    print("\n\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    for layer_idx, gradient in enumerate(final_gradients):
        print(f"\t\tFinal regularized gradients of Theta{layer_idx + 1}:")
        for row in gradient:
            print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))

if __name__ == "__main__":
    case = None # Set to 1 for the first test case, 2 for the second test case, or None to skip test cases
    mode = 2

    dataset_path = "loan.csv" # Change this for using different datasets
    X, y = preprocessDataset(dataset_path)

    indices = np.random.permutation(X.shape[0]) 
    X = X[indices] 
    y = y[indices] 

    layer_sizes = [X.shape[1], 16, 1]
    alpha_learning_rate=0.01
    lambda_regulation=0.1
    max_iterations=340
    batch_size=32
    # epsilon=0.00001
    k=5

    if case is not None:
        testCase(case)
    elif mode == 1:
        splits = stratifiedKFoldCrossValidation(X, y, k=k)

        fold_metrics = []
        total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0


        for fold, (X_train, y_train, X_test, y_test) in enumerate(splits):
            print(f"\nProcessing Fold {fold + 1}/{k}")

            weights = generateInitialWeights(layer_sizes)

            trained_weights, gradients_regularizationients = trainModel(
                X_train, y_train,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_regulation=lambda_regulation,
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
        dataset_split_ratio = 0.8 
        dataset_split_index = int(dataset_split_ratio * X.shape[0])
        X_train, X_test = X[:dataset_split_index], X[dataset_split_index:]
        y_train, y_test = y[:dataset_split_index], y[dataset_split_index:]

        training_sizes = []
        test_costs = []

        weights = generateInitialWeights(layer_sizes)

        for samples in range(10, X_train.shape[0] + 1, 10): 
            print(f"\nTraining with {samples} samples.")

            X_train_subset = X_train[:samples]
            y_train_subset = y_train[:samples]

            trained_weights, gradients_regularizationients = trainModel(
                X_train_subset, y_train_subset,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_regulation=lambda_regulation,
                max_iterations=max_iterations,
                batch_size=batch_size,
                # epsilon=epsilon
            )

            test_cost = calculateCost(X_test, y_test, trained_weights, lambda_regulation)
            print(f"Test Cost (J) with {samples} training samples: {test_cost:.5f}")

            training_sizes.append(samples)
            test_costs.append(test_cost)
        
        dataset_title = (dataset_path.split("/")[-1].split(".")[0]).capitalize() 

        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, test_costs, label="Cost (J)", linewidth=2)
        plt.title(f"Learning Curve - NN: {layer_sizes} - {dataset_title} Dataset - Lambda: {lambda_regulation} - Alpha: {alpha_learning_rate} - Batch Size: {batch_size}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Cost Function (J)")
        plt.legend()
        plt.show()