import os
import pandas as pd
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), 'breast-cancer.csv')
df = pd.read_csv(data_path)



df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': -1 })



x = df.drop(columns=['id', 'diagnosis']).apply(pd.to_numeric, errors='coerce')
x = x.fillna(0)
y = df['diagnosis']

X = x.head(400).to_numpy().astype(float)
Y = y.head(400).to_numpy()

x_test = x.tail(169).to_numpy().astype(float)
y_test = y.tail(169).to_numpy()


n_sample, m_feature = X.shape

lr = 0.00001
step = 1000
lambda_para = 0.01

w = np.zeros(m_feature)
b = 0

def learn(lambda_para,lr,step,X,Y,w,b):
    for _ in range(step):
        for idx, x_i in enumerate(X):
            condition = Y[idx]*(np.dot(x_i,w)-b)
            if condition >= 1:
                w -= lr*(2*lambda_para*w)
            else:
                w -= lr*(2*lambda_para*w - np.dot(x_i, Y[idx]) )
                b += lr*Y[idx]
    return w, b



def accuracy(w, b, X_test, y_test):
    pred = np.sign(np.dot(X_test, w) - b)
    acc = np.mean(pred == y_test)
    print('accuracy:', acc)

w, b = learn(lambda_para,lr,step,X,Y,w,b)

accuracy(w,b,x_test,y_test)



     

