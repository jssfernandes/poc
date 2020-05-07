# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('bases/housing.csv')
df = df.drop(columns=['ocean_proximity'])
df.dropna(inplace=True)

x = df.iloc[:, :8]
y = df.iloc[:, -1]

print(df.info())

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

print(regressor.predict([[-122, 37, 41, 880, 129, 322, 126, 8]]))
'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print(regressor.predict([[-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]]))
'''