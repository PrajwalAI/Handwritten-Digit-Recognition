import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shade=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epoches=3)

model.save('digit_recognition.model')

# testing the Data
model.tf.keras.models.load_model('digit_recognition.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# making prediction on new inputs

img_num = 1
while os.path.isfile(f"folder1/digits{img_num}.png"):        # add the images in folder1
    try:
        img = cv2.imread(f"folder1/digits{img_num}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit probably is a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        img_num += 1
