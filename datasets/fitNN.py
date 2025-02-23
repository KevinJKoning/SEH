import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Data as a multi-line string
with open('/workspaces/SEH/datasets/sampled_diamonds.csv', 'r') as file:
    data_str = file.read()

# Load only the carat (column 1) and price (column 7) data.
data = np.genfromtxt(StringIO(data_str), delimiter=",", skip_header=1, usecols=(1, 7))
X = data[:, 0].reshape(-1, 1)  # input: carat
y = data[:, 1].reshape(-1, 1)  # target: price


# Normalize the input and output
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / y_std

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_deriv(x):
    sig = logistic(x)
    return sig * (1 - sig)

# Neural network with two hidden layers, biases, and improved initialization.
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation="logistic"):
        self.activation = activation.lower()  # "relu" or "logistic"
        # Weight initialization (using He initialization for ReLU, similar idea for logistic)
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.b3 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.activation == "relu":
            self.A1 = relu(self.Z1)
        else:
            self.A1 = logistic(self.Z1)
            
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        if self.activation == "relu":
            self.A2 = relu(self.Z2)
        else:
            self.A2 = logistic(self.Z2)
            
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        # Linear output (for regression)
        return self.Z3
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y_true, y_pred, learning_rate):
        m = y_true.shape[0]
        dZ3 = (2.0 / m) * (y_pred - y_true)
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, self.W3.T)
        if self.activation == "relu":
            dZ2 = dA2 * relu_deriv(self.Z2)
        else:
            dZ2 = dA2 * logistic_deriv(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        if self.activation == "relu":
            dZ1 = dA1 * relu_deriv(self.Z1)
        else:
            dZ1 = dA1 * logistic_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            self.backward(X, y, y_pred, learning_rate)
            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        return losses

# Hyperparameters
hidden_neurons = 4     # Number of neurons in each hidden layer
activation = "logistic"  # Choose "relu" or "logistic"
epochs = 10000
learning_rate = 0.01

# Create and train the network
nn = NeuralNetwork(input_size=1, hidden_size1=hidden_neurons, hidden_size2=hidden_neurons,
                   output_size=1, activation=activation)
losses = nn.train(X_norm, y_norm, epochs, learning_rate)

# Generate predictions (denormalize the output)
y_pred_norm = nn.forward(X_norm)
y_pred = y_pred_norm * y_std + y_mean

# Plot training loss and predictions vs actual data
plt.figure(figsize=(12, 5))

# Training loss plot
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

# Prediction plot
plt.subplot(1, 2, 2)
plt.scatter(X, y, color="blue", label="Actual Price")
plt.scatter(X, y_pred, color="red", label="Predicted Price")
plt.title("Actual vs. Predicted Diamond Price")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.legend()

plt.tight_layout()
plt.show()
