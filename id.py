import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array([[-1, -1], [10, 12], [4, 3], [2, -4]], dtype=np.float)
y = np.array([1, 1, 2, 2])

scaler = StandardScaler()

"""
print(X)
print(f'the whole matrix X | sum {np.sum(X)} | mean {np.mean(X)} | standard deviation {np.std(X)}')
print(f'column1 | sum {np.sum(X[:, 0])} | mean {np.mean(X[:, 0])} | standard deviation {np.std(X[:, 0])}')
print(f'column2 | sum {np.sum(X[:, -1])} | mean {np.mean(X[:, -1])} | standard deviation {np.std(X[:, -1])}')
"""
Y = X.copy()
x_mean = np.mean(X)
x_std = np.std(X)
X = scaler.fit_transform(X)

y0_mean = np.mean(Y[:, 0])
y0_std = np.std(Y[:, 0])
y1_mean = np.mean(Y[:, -1])
y1_std = np.std(Y[:, -1])
print(y0_mean, y0_std, y1_mean, y1_std)

#Y = (Y - x_mean) / x_std

print(np.sum(Y, axis=1))
Y = (Y - np.mean(Y, axis = 0)) / np.std(Y, axis = 0)

"""
for j in range(2):
    for i in range(len(Y)):
        if j == 0:
            Y[i][j] = (Y[i][j] - y0_mean ) / y0_std
        else:
            Y[i][j] = (Y[i][j] - y1_mean) / y1_std
"""
print(np.mean(X), np.std(X))
print(np.mean(Y), np.std(Y))
print(X)
print(Y)