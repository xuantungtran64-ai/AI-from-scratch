import os
from dec_tree import Dec_tree as d_tr
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = os.path.join(os.path.dirname(__file__), 'Social_Network_Ads.csv')
df = pd.read_csv(data_path)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

x = df.drop(columns = ['User ID','Purchased']).values
y = df['Purchased'].values

X_train,X_test, Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)




class random_forest:
    def __init__(self,n_tree = 10 , max_depth = 10,min_samp_split = 2,n_features =None):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_samp_split = min_samp_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X,y):
        self.trees = []
        for _ in range(self.n_tree):
            tree = d_tr(min_samp_split = self.min_samp_split, max_depth = self.max_depth)
            X_sample , y_sample = self.bootstrap_sample(X,y)
            tree.fit(X_sample , y_sample)
            self.trees.append(tree)
    
    def bootstrap_sample(self,X,y):
        n_sample = X .shape[0]
        idxs = np.random.choice(n_sample,n_sample, replace = True)
        return X[idxs], y[idxs]
    
    def most_common_lables(self,y):
        counter = Counter(y)
        most_Common =  counter.most_common(1)[0][0]
        return most_Common
        
    def predict(self,X):
        pred = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(pred,0,1)
        preditions = np.array([self.most_common_lables(pre) for pre in tree_preds])
        return preditions
    



rd = random_forest()
rd.fit(X_train,Y_train)
predict = rd.predict(X_test)
accuracy = np.mean(predict == Y_test)*100
print(f"accuracy: {accuracy:.2f}%")


