import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Step 3: Load the MNIST dataset
# The MNIST dataset is available directly in Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 4: Preprocess the data
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 5: Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into vectors of size 784
    layers.Dense(128, activation='relu'),  # First dense layer with 128 units and ReLU activation
    layers.Dropout(0.2),                   # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax') # Output layer with 10 units (one per digit) and softmax activation
])

# Step 6: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train the model
model.fit(x_train, y_train, epochs=5)

# Step 8: Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc}")

# Step 9: (Optional) Visualize some predictions
predictions = model.predict(x_test)

# Plot a few test images with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(f"Pred: {predictions[i].argmax()} (True: {y_test[i]})")
plt.show()
