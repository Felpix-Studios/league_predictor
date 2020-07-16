# imports!
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

defaults = [tf.int64] + [tf.int32] * 12 + [tf.float32] + [tf.int32] * 5 + [tf.float32] * 2 +[tf.int32] * 11 + [tf.float32]+ [tf.int32] * 5 + [tf.float32] *2
dataset = tf.data.experimental.CsvDataset("dataset.csv",defaults)

datan=np.genfromtxt('dataset.csv',delimiter=',')

col_names = ['blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths','blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','blueTotalGold','blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled','blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','blueCSPerMin','blueGoldPerMin','redWardsPlaced','redWardsDestroyed','redFirstBlood','redKills','redDeaths','redAssists','redEliteMonsters','redDragons','redHeralds','redTowersDestroyed','redTotalGold','redAvgLevel','redTotalExperience','redTotalMinionsKilled','redTotalJungleMinionsKilled','redGoldDiff','redExperienceDiff','redCSPerMin','redGoldPerMin']
def _parse_csv_row(*vals):
    winner=tf.convert_to_tensor(vals[1])
    feat_vals=tf.convert_to_tensor(vals[2:])

    features=dict(zip(col_names,feat_vals))
    return features

dataset = dataset.map(_parse_csv_row).batch(64)

print(list(dataset.take(1)))