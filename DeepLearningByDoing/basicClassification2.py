import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

print(tf.__version__)

digits_data = datasets.load_digits()

train_x, test_x, train_y, test_y = train_test_split(digits_data.data, digits_data.target, test_size=.4, random_state=0)

print(train_x.shape)
print(train_y.shape)

plt.figure()
plt.imshow(digits_data.images[0])
plt.colorbar()
plt.grid(False)


train_x = train_x / 8.0
test_x = test_x / 8.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(digits_data.images[i], cmap=plt.cm.binary)
    plt.xlabel(digits_data.target[i])
plt.show()

model = keras.Sequential([
    keras.layers.Dense(64),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=15)

test_los, test_acc = model.evaluate(test_x, test_y)
print("Test accuracy is : {}".format(test_acc))



