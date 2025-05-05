import numpy as np
from NeuralNetworkArchitecture import backwardPropagation, calculateCost

def gradientCheck(X, y, weights, lambda_regularization, epsilon):
    numerical_gradients = []
    backprop_gradients = backwardPropagation(X, y, weights, lambda_regularization)

    for l, theta in enumerate(weights):
        num_grad = np.zeros_like(theta) 
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[i, j] += epsilon
                theta_minus[i, j] -= epsilon

                weights_plus = weights.copy()
                weights_minus = weights.copy()
                weights_plus[l] = theta_plus
                weights_minus[l] = theta_minus

                J_plus = calculateCost(X, y, weights_plus, lambda_regularization)
                J_minus = calculateCost(X, y, weights_minus, lambda_regularization)

                num_grad[i, j] = (J_plus - J_minus) / (2 * epsilon)

        numerical_gradients.append(num_grad)

        print(f"\nLayer {l + 1} Gradients:")
        print(f"Numerical Gradient:\n{num_grad}")
        print(f"Backpropagation Gradient:\n{backprop_gradients[l]}")

        diff = np.linalg.norm(num_grad - backprop_gradients[l]) / (np.linalg.norm(num_grad) + np.linalg.norm(backprop_gradients[l]))
        print(f"Relative Difference: {diff:.5f}")

        if diff > 1e-4:
            print("Gradients did not match.")
        else:
            print("Gradients matched.")

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
        lambda_regularization = 0.0

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
        lambda_regularization = 0.25

    X_train = np.array([x for x, _ in training_data])
    y_train = np.array([y for _, y in training_data])

    epsilon_values = [0.1, 1e-6]
    for epsilon in epsilon_values:
        print(f"\nGradient Check with {epsilon}.")
        gradientCheck(X_train, y_train, weights_list, lambda_regularization, epsilon)

if __name__ == "__main__":
    case = 1 # Change to 1 for first test case or 2 for second test case
    testCase(case)