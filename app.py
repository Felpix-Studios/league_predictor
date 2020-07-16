# imports!
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

fig = plt.figure(figsize=(18,6))

df.blueKills.value_counts().plot(kind="bar",alpha=0.5)

plt.show()