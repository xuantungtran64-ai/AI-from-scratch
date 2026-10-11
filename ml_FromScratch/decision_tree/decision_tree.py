import os
from tree_node import Node     
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split


data_path = os.path.join(os.path.dirname(__file__), 'Social_Network_Ads.csv')
df = pd.read_csv(data_path)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

x = df.drop(columns = ['User ID','Purchased']).values
y = df['Purchased'].values

X_train,X_test, Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)


class Dec_tree:
    def __init__(self, min_samp_split = 2, max_depth = 2):
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth

    def build_tree(self,data, curr_depth = 0):
        X,y = data[:, :-1], data[:,-1]
        samples , features = X.shape

        if samples >= self.min_samp_split and curr_depth <= self.max_depth:
            best_split = self.best_split(data, features)

            if best_split['info_gained'] > 0:
                left_node = self.build_tree(best_split['left_dataset'],curr_depth + 1)
                right_node = self.build_tree(best_split['right_dataset'],curr_depth + 1)

                return Node(best_split['feature_idx'], best_split['threshold'], best_split['info_gained'],left_node,right_node)

        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value = leaf_value)
        
    def best_split(self,data,features):
        best_split = {'feature_idx': None, 'threshold':None,'info_gained' : -1, 'left_dataset' : None, 'right_dataset' : None}

        for feature_idx in range(features):
            features_value = data[:,feature_idx]
            thresholds = np.unique(features_value)

            for threshold in thresholds:
                left_dataset , right_dataset = self.split(data,feature_idx, threshold)

                if len(left_dataset) and len(right_dataset):
                    parent_y, left_y, right_y = data[:,-1], left_dataset[:,-1],right_dataset[:,-1]

                    info_gained = self.information_gain(parent_y,left_y,right_y)

                    if info_gained > best_split['info_gained']:
                        best_split['feature_idx'] = feature_idx
                        best_split['threshold'] = threshold
                        best_split['info_gained'] = info_gained
                        best_split['left_dataset'] = left_dataset
                        best_split['right_dataset'] = right_dataset

        return best_split
    



    def split(self,data,feature_idx, threshold):
        left_dataset = np.array([row for row in data if row[feature_idx] <= threshold])
        right_dataset = np.array([row for row in data if row[feature_idx] > threshold])

        return left_dataset, right_dataset
    

    def information_gain(self,parent_y,left_y, right_y):
        left_w = len(left_y) / len(parent_y)

        right_w = len(right_y) / len(parent_y)

        info_gain = self.entropy(parent_y) - (left_w*self.entropy(left_y)+right_w*self.entropy(right_y))

        return info_gain
    
    def entropy(self,y):
        entropy = 0

        class_lables = np.unique(y)
        for class_lable in class_lables:
            p = len(y[y==class_lable]) / len(y)

            entropy += -p*np.log2(p) 
        return entropy
    

    def fit(self,X,y):
        data = np.concatenate([X,y.reshape(-1,1)],axis=1)
        self.root = self.build_tree(data)

    def predict(self,X):
        predict = [self.predict_class(row,self.root) for row in X]
        return predict
    
    def predict_class(self,row,node):
        if node.value != None:
            return node.value

        feature_val = row[node.feature_idx]
        if feature_val <= node.threshold:
            return self.predict_class(row,node.left)
        else:
            return self.predict_class(row,node.right)
        
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.value is not None:
            print(indent + "Leaf:", node.value)
            return

        print(indent + f"[X{node.feature_idx} <= {node.threshold}]")
        self.print_tree(node.left, indent + "  ")
        self.print_tree(node.right, indent + "  ")
            

dt = Dec_tree(min_samp_split = 2, max_depth = 2)
dt.fit(X_train,Y_train)
predict = dt.predict(X_test)
accuracy = np.mean(predict == Y_test)*100
print(f"accuracy: {accuracy:.2f}%")
dt.print_tree()

