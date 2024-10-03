from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc}")

predictions = model.predict(x_test)
"""
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(f"Pred: {predictions[i].argmax()} (True: {y_test[i]})")
plt.show()
"""

def predict_number(img):
    global model
    if len(img.shape) == 3 and img.shape[2] == 3:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(img, (28, 28))

    img_normalized = img_resized / 255.0

    img_reshaped = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_reshaped)

    predicted_digit = np.argmax(predictions)

    return predicted_digit