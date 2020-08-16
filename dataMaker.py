# imports!
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, random, pickle


remove_cols = ["gameId","blueTotalMinionsKilled","redTotalMinionsKilled","blueTotalMinionsKilled","redTotalMinionsKilled","blueTotalJungleMinionsKilled","redTotalJungleMinionsKilled","blueGoldDiff","redGoldDiff","blueExperienceDiff","redExperienceDiff","blueCSPerMin","redCSPerMin","blueGoldPerMin","redGoldPerMin"]
data = pd.read_csv("dataset.csv")
data = data.drop(remove_cols, axis=1)
msk = np.random.rand(len(data)) < 0.8
x_train = data[msk]
x_test = data[~msk]

# sns.pairplot(x_train[["blueWins","blueKills", "redKills", "blueTowersDestroyed", "redTowersDestroyed"]], diag_kind="kde")

#sns.pairplot(x_train[["blueWins","blueKills","blueAssists","blueTotalExperience","redTotalExperience","redAssists", "redKills"]], size=3, palette='Set1')
print(x_train.corr(method = 'pearson'))

plt.figure(figsize=(16, 12))
sns.heatmap(x_train.corr(), cmap='YlGnBu', annot=True, fmt='.2f', vmin=0);

train_stats = x_train.describe()
train_stats.pop("blueWins")
train_stats=train_stats.transpose()



y_train = x_train.pop('blueWins')
y_test = x_test.pop('blueWins')

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# Normalizing data
def norm(x):
    return(x-train_stats['mean']/train_stats['std'])

norm_x_train = norm(x_train)
norm_x_test = norm(x_test)

pickle_out = open("x_train.pickle","wb")
pickle.dump(norm_x_train,pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(norm_x_test,pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()

plt.show()