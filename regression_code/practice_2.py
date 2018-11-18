import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso , Ridge
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('8.Advertising.csv')
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
model = Lasso()
alpha_can = np.logspace(-3, 2, 10)
lasso_model = GridSearchCV(model, param_grid={'alpha':alpha_can}, cv=5)
lasso_model.fit(x, y)
y_hat = lasso_model.predict(x_test)
mse = np.average((y_hat - y_test) ** 2)
rmse = np.sqrt(mse)
print(mse, rmse)
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='best')
plt.grid()
plt.show()

