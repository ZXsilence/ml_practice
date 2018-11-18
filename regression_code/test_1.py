import csv
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('8.Advertising.csv')
x = data[['TV','Radio', 'Newspaper']]
y = data['Sales']
# plt.plot(data['TV'], y, 'ro', label='TV')
# plt.plot(data['Radio'], y, 'g^', label='Radio')
# plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
# plt.legend(loc='best')
# plt.grid()
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(np.array(x_train), np.array(y_train))
y_hat = linreg.predict(np.array(x_test))
print(y_hat.shape)
print(y_test.shape)
mse = np.average((y_hat - np.array(y_test)) ** 2 )
rmse = np.sqrt(mse)
print(mse, rmse)
t = np.arange(len(x_test))
print(t)
print(type(t))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()