import os
import pandas as pd
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), 'framingham.csv')
df = pd.read_csv(data_path)

#df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

dt_train = df.head(4000)
dt_test = df.tail(230)

X_train = dt_train[['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke',
                    'prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']].to_numpy(dtype=float)
Y_train = dt_train['TenYearCHD'].to_numpy(dtype=float)

X_test = dt_test[['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke',
                  'prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']].to_numpy(dtype=float)
Y_test = dt_test['TenYearCHD'].to_numpy(dtype=float)



def sigmoid(z):
    return 1/(1+np.exp(-z))

def grad(O,X,y):
    m = y.size #instances
    return (X.T @ (sigmoid(X @ O) - y ))/m

def grad_desc(X,y,r = 0.1, mult = 100000, tol = 1e-7):
    X_b = np.c_[np.ones((X.shape[0],1)),X]

    O = np.zeros(X_b.shape[1])

    for i in range(mult):
        gradient = grad(O,X_b,y)
        O -= r*gradient
        if np.linalg.norm(gradient) < tol:
            break
    return O

def predict_prob(X,O):
    X_b = np.c_[np.ones((X.shape[0],1)),X]
    return sigmoid(X_b @ O)

def predict(X,O,T = 0.5):
    return (predict_prob(X,O)>= T).astype(int)

O_hat = grad_desc(X_train,Y_train)

y_pred = predict(X_train, O_hat)

accuracy = (y_pred == Y_train).mean()
print("Accuracy:", accuracy)







