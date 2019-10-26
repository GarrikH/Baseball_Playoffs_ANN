#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import tensorflow as tf

filename = "baseball.csv"
baseball_df = read_csv(filename)

# baseball_df['WS_Champ'] = np.where(baseball_df['RankPlayoffs']==1.0 , 1, 0)
baseball_df['RSPG'] = baseball_df['RS'] / baseball_df['G']
baseball_df['RAPG'] = baseball_df['RA'] /baseball_df['G']
baseball_df['RDIFF'] = baseball_df['RSPG'] - baseball_df['RAPG']
baseball_df['W'] = baseball_df['W'] / 100
baseball_df_nostrings = baseball_df[["W", "OBP", "SLG", "BA", "OOBP", "OSLG", "RSPG", "RAPG", "RDIFF", "Playoffs"]]
baseball_df_nostrings = baseball_df_nostrings.astype(float)
baseball_df_nostrings['Playoffs'] = baseball_df_nostrings['Playoffs'].astype(int)
baseball_df_nostrings['OOBP'].fillna(baseball_df_nostrings['OOBP'].mean(), inplace=True)
baseball_df_nostrings['OSLG'].fillna(baseball_df_nostrings['OSLG'].mean(), inplace=True) 

bb_array = baseball_df_nostrings.values
x = bb_array[:, 0:8]
y = bb_array[:, 9]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=9)

model = Sequential()
model.add(Dense(9, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 1000, batch_size=32)

# Baseline test set pred: Testing Set accuracy: 77.27%
scores = model.evaluate(x_test, y_test)
print("Testing Set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_hats = model.predict(x)
baseball_df['y_hats'] = y_hats
baseball_df['y_hats'] = np.where(baseball_df['y_hats'] >= 0.5, 1, 0)

baseball_df.to_csv(r'BaseballWithPreds.csv', index=False)