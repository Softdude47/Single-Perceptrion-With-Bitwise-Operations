import numpy as np
from mas_lib.nn.perceptron import Perceptron
from mas_lib.nn.neuralnetwork import NeuralNetwork

# initialize the AND data array and labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
EPOCHS = 2000

# Using SINGLE LAYER PERCEPTRON
# initialize the perceptron network
P = Perceptron(X.shape[1])

# training the perceptron on the AND data
print("[INFO]: Training Single Layer Perceptron")
P.fit(X, y, epochs=20)

# confirms wether or not our model was able to
# learn patterns on the training data
print("\n[INFO]: Testing Perceptron")
for (data, label) in zip(X, y):
    # make predictions and see compare them with the
    # actual labels
    predicted = P.predict(data)
    print(f"[INFO]: data={data} ground-truth= {label} predicted= {predicted}")


# Neural Network
print("\n\n[[INFO]: Using Multi Layer Perceptron")
MLP = NeuralNetwork([2, 2, 1], 0.5)

# training the model
MLP.fit(X, y, epochs=EPOCHS, verbose=500)

# confirms wether or not our model was able to
# learn patterns on the training data
print("\nINFO]: Testing Multi Layer Perceptron")
for (data, label) in zip(X, y):
    # make predictions and see compare them with the
    # actual labels
    predicted = MLP.predict(data)[0][0]
    step = 1 if predicted > 0.5 else 0
    print(f"[INFO]: data= {data} ground-truth= {label} predicted= {predicted} step= {step}")
