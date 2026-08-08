import time
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)

y = np.array([
    0,
    1,
    1,
    0
])


# Same architecture:
# C++: 2 -> 4 -> 1
#
# tanh -> sigmoid
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="sgd",
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=42
)


# -----------------------------
# Training
# -----------------------------

start = time.perf_counter()

model.fit(X, y)

end = time.perf_counter()

training_time = end - start


# -----------------------------
# Prediction
# -----------------------------

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)


# -----------------------------
# Results
# -----------------------------

print("\n===========================")
print("Python Neural Network")
print("===========================")

print(f"Training Time: {training_time:.6f} seconds")

print("\nPredictions:")
print(predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")