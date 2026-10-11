import os
import pandas as pd
import matplotlib.pyplot as plt

train_path = os.path.join(os.path.dirname(__file__), 'train.csv')
df = pd.read_csv(train_path)
#print(df)

'''plt.scatter(df.x,df.y)
plt.show()'''

def loss_function (m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m*x+b))**2
    return total_error / float(len(points))

def grad_desc(m_now,b_now,points,r):
    m_grad = 0
    b_grad = 0

    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_grad += -(2/n)*x*(y-(m_now*x+b_now))
        b_grad += -(2/n)*(y-(m_now*x+b_now))

    m = m_now - r*m_grad
    b = b_now - r*b_grad
    return m,b

m = 0
b = 0
r = 0.0001
epochs = 1000
for i in range(epochs):
    m,b = grad_desc(m,b,df,r)

print(m,b)
plt.scatter(df.x,df.y)
plt.plot((list(range(0,100))),[m*x+b for x in range (0,100)], color = 'red')
plt.show()

'''def linear_regression(x):
    y = m*x + b
    return y'''

test_path = os.path.join(os.path.dirname(__file__), 'test.csv')
d_test = pd.read_csv(test_path)

plt.scatter(d_test.x,d_test.y)
plt.plot((list(range(0,100))),[m*x+b for x in range (0,100)], color = 'red')
plt.show()