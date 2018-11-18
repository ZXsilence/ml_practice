import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

path = '8.iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
print(data)
x, y = np.split(data, (4,), axis=1)
print(x)
print(y)
x = x[:,:2]
# x = StandardScaler().fit_transform(x)

# print(x)
# lr = LogisticRegression()

lr = Pipeline([('a',StandardScaler()),('b',LogisticRegression())])
# print(type(y.ravel()))

lr.fit(x, y)
y_hat = lr.predict(x)
result = y_hat == y.reshape(-1)
print(np.mean(result))
N, M = 500, 500
x1_min, x1_max = x[:,0].min(), x[:, 0].max()
x2_min, x2_max = x[:,1].min(), x[:, 1].max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)
print(x1.shape)
print(t1)
x_test = np.stack((x1.flat, x2.flat), axis=1)
# print(x1, x2)
print(x_test.shape)
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_hat = lr.predict(x_test)
y_hat = y_hat.reshape(x1.shape)
print(x1_min,x1_max)
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), edgecolors='face', s=199, cmap=cm_dark)
plt.xlabel('petal length')
plt.ylabel('petal width')

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()
plt.show()