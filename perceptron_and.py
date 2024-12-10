import numpy as np
from mas_lib.nn.perceptron import Perceptron

# initialize the AND data array and labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# initialize the perceptron network
P = Perceptron(X.shape[1])

# training the perceptron on the AND data
print("[INFO]: Training Perceptron")
P.fit(X, y, epochs=20)

# confirms wether or not our model was able to
# learn patterns on the training data
print("[INFO]: Testing Perceptron")
for (data, label) in zip(X, y):
    # make predictions and see compare them with the
    # actual labels
    predicted = P.predict(data)
    print(f"[INFO]: predicted= {predicted}, ground-truth= {label}")