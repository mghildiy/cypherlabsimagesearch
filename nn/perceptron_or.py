import numpy as np
from nn.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

print(X.shape)
print(y.shape)

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))