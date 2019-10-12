import numpy as np
import cv2

labels = ["dog", "cat", "panda"]
np.random.seed(1)

# random initialization of weight matrix and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread("beagle.png")
print(orig.shape)
image = cv2.resize(orig, (32,32)).flatten()
print(image.shape)

# output scores = W * input + b
scores = W.dot(image) + b
print(scores)

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)