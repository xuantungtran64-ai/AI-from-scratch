import os
import pandas as pd
import numpy as np


data_path = os.path.join(os.path.dirname(__file__), 'breast-cancer.csv')
df = pd.read_csv(data_path)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0 })

x_df = df.drop(columns=['id', 'diagnosis']).apply(pd.to_numeric, errors='coerce')
x_df = x_df.fillna(0)
x_df = (x_df - x_df.min()) / (x_df.max() - x_df.min())

y_df = df['diagnosis']



h = int(0.8*df.shape[0])
t = df.shape[0] - h

x = x_df.head(h).reset_index(drop=True)
y = y_df.head(h).reset_index(drop=True)

x_test = x_df.tail(t).reset_index(drop=True)
y_test = y_df.tail(t).reset_index(drop=True)



def distance(a,b):
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2))
    
def to_k(df):
    return int((df.shape[0])**0.5)

def pred(x,y,x_t,k):
    y_pred = []
    for i in range(len(x_t)):
        point_a = x_t.iloc[i].to_numpy()
        distances = []

        for j in range(len(x)):
            point_b = x.iloc[j].to_numpy()
            d = distance(point_a,point_b)
            distances.append([d, y.iloc[j]])
        distances.sort(key=lambda sub: sub[0])
        votes = [label for (_, label) in distances[:k]]
        count_1 = votes.count(1)
        count_0 = votes.count(0)
        
        y_pred.append(1 if count_1 > count_0 else 0)
    return y_pred


def accuracy(y_pred,y_t):
    y_pred = np.array(y_pred)
    return round(float(np.mean(y_pred == y_t)),2)





k = 1
best_acc = 0
best_k = k
no_improve = 0
max_no_improve = 10

while True:
    y_pred = pred(x, y, x_test, k)
    acc = accuracy(y_pred, y_test)

    print(f'k={k}, accuracy={acc}')

    if acc > best_acc:
        best_acc = acc
        best_k = k
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= max_no_improve:
        break

    k += 1

print(f'best k={best_k}, best accuracy={best_acc}')


            