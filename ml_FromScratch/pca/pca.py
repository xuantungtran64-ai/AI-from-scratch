import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class pca:
    def __init__(self, k=2):
        self.k = k
        self.comp = None
        self.mean = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        cov = np.cov(x - self.mean, rowvar=False)
        val, vec = np.linalg.eigh(cov)
        idx = np.argsort(val)[::-1]
        self.comp = vec[:, idx[:self.k]]
        return self

    def transform(self, x):
        return np.dot(x - self.mean, self.comp)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

def PCA(x, k=2):
    return pca(k).fit_transform(x)

f = 'iris.csv' if os.path.exists('iris.csv') else os.path.join(os.path.dirname(__file__), 'iris.csv')
df = pd.read_csv(f)

x = df.iloc[:, 1:-1].to_numpy().astype(float)
y = df.iloc[:, -1].to_numpy()

p = pca(2)
x_proj = p.fit_transform(x)

print('original:', x.shape)
print('projected:', x_proj.shape)

for s in np.unique(y):
    plt.scatter(x_proj[y == s, 0], x_proj[y == s, 1], label=str(s).lower())

plt.title('pca - iris')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.grid(True)
plt.show()