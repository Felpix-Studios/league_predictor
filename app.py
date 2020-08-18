# imports!
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, random, pickle

fig, ax = plt.subplots()

CATEGORIES = ["Lose","Win"]

x_train = pickle.load(open("x_train.pickle","rb"))
y_train = pickle.load(open("y_train.pickle","rb"))
x_test = pickle.load(open("x_test.pickle","rb"))
y_test = pickle.load(open("y_test.pickle","rb"))

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# sns.pairplot(x_train[["blueKills", "redKills", "blueTowersDestroyed", "redTowersDestroyed"]], diag_kind="kde")
# plt.show()

def build_model():
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation = "relu", input_shape=[len(x_train.keys())]),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(1)
        #tf.keras.layers.Dense(64, activation = "relu"),
        #tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'binary_crossentropy', optimizer="adam",metrics = "accuracy")

    return model

model = build_model()

model.summary()

model.fit(x_train,y_train,epochs=500, verbose = 0)

train_loss,train_acc = model.evaluate(x_train,y_train)
print(train_loss,train_acc)

test_loss,test_acc = model.evaluate(x_test,y_test)
print(test_loss,test_acc)

prediction = model.predict(x_test)

print(prediction)
print(prediction[0])
print(prediction[0][0])
print(y_test.head())