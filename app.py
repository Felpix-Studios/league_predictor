# imports!
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

defaults = [tf.int64] + [tf.int32] * 12 + [tf.float32] + [tf.int32] * 5 + [tf.float32] * 2 +[tf.int32] * 11 + [tf.float32]+ [tf.int32] * 5 + [tf.float32] *2

fig, ax = plt.subplots()

data = pd.read_csv("dataset.csv")
msk = np.random.rand(len(data)) < 0.8
x_train = data[msk]
x_test = data[~msk]
# y_train = x_train.pop('blueWins')

print(len(x_train))
print(len(x_test))
x_train.blueKills.hist(bins=20)
ax.set_xlabel('Blue Kills')
ax.set_ylabel('Occurrences')

plt.show()