# imports!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, random, pickle


remove_cols = ["gameId","blueTotalMinionsKilled","redTotalMinionsKilled"]
data = pd.read_csv("dataset.csv")
data = data.drop(remove_cols, axis=1)
msk = np.random.rand(len(data)) < 0.8
x_train = data[msk]
x_test = data[~msk]

y_train = x_train.pop('blueWins')
y_test = x_test.pop('blueWins')

x_train.head()
y_train.head()
x_test.head()
y_test.head()
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train,pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test,pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()