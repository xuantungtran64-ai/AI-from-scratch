import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle 


class LayerDense:

    def __init__(self,n_input,n_neuron):
        self.weights = 0.1*np.random.randn(n_input,n_neuron)
        self.biases = np.zeros((1,n_neuron))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self, dvalues):
        self.d_biases = np.sum(dvalues,axis=0,keepdims=True)
        self.d_weights = np.dot(self.inputs.T,dvalues)
        self.d_inputs = np.dot(dvalues,self.weights.T)


class relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.d_inputs = np.where(self.inputs <= 0, 0, dvalues)


class sigmoid:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = 1/ (1+np.exp(-inputs))

    def backward(self,dvalues):
        self.d_inputs = dvalues * (self.output) * (1-self.output)


class softmax:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.output = exp_values / np.sum(exp_values,axis = 1, keepdims= True)

    def backward(self,dvalues):
        diagonal = np.diag(self.output)[:None]
        pad = np.zeros(diagonal.shape)
        sii = np.hstack((np.diagflat(diagonal),pad))
        self.d_inputs = sii - (diagonal * self.output)
        #in progress, let skip it as it's not needed


class Loss:
    def forward(self,y_pred,y_true):
        self.batch_size = y_pred.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true
        p = y_pred[y_true == 1]
        p = np.clip(p,0+1e-9,1 - 1e-9)
        self.loss =  - np.average(np.log(p))
    
    def backward(self):
        y_pred = np.clip(self.y_pred,0+1e-9,1 - 1e-9)
        self.d_loss =  - self.y_true/ (y_pred*self.batch_size)

    def backward_with_softmax(self):
        self.d_inputs = (self.y_pred - self.y_true) / self.batch_size


class optimizer:
    def __init__(self,lr = 0.001):
        self.lr = lr

    def OptimizerGD(self,layer):
        layer.weights -= self.lr*layer.d_weights
        layer.biases -= self.lr*layer.d_biases




def OneHotMatrix(labels, num_classes):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    y_true = np.zeros((len(labels), num_classes))
    y_true[np.arange(len(labels)), labels] = 1
    return y_true


def accuracy(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)

    if len(y_true.shape) == 2:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true

    return np.mean(y_pred_labels == y_true_labels)


class Model:
    def __init__(self,layers):
        self.layers = layers

    def save(self,filename):
        with open (filename,'wb') as f:
            pickle.dump(self,f)

    @staticmethod
    def load(filename):
        with open(filename,'rb') as f:
            return pickle.load(f)
        
    def forward(self,X):
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output



data_path = os.path.join(os.path.dirname(__file__), 'breast-cancer.csv')
df = pd.read_csv(data_path)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0 })

x_df = df.drop(columns=['id', 'diagnosis']).apply(pd.to_numeric, errors='coerce')
X = x_df.fillna(0)

y = df['diagnosis']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

#CONFIG
epoches = 1000
batch_size = 64
features = X_train.shape[1]
patience  = 10

#training
ld1 = LayerDense(features, 32 )
activation1 = relu()

ld2 = LayerDense(32,16)
activation2 = relu()

ld3 = LayerDense(16,2)
activation3 = softmax()

L = Loss()

opt = optimizer()

epoch_loss = 0
epoch_acc = 0
num_batches = 0


pat = 0

lowest_Loss = np.inf
num_classes = len(np.unique(y))
y_val_oh = OneHotMatrix(y_val,num_classes)

for epoch in range(epoches):

    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]


    for b in range(0,len(X_train_shuffled),batch_size):
        X_batch = X_train_shuffled[b:b+batch_size]
        y_batch = OneHotMatrix(y_train_shuffled[b:b+batch_size],num_classes)


        ld1.forward(X_batch)
        activation1.forward(ld1.output)

        ld2.forward(activation1.output)
        activation2.forward(ld2.output)

        ld3.forward(activation2.output)
        activation3.forward(ld3.output)

        L.forward(activation3.output,y_batch)

        acc = accuracy(activation3.output, y_batch)

        L.backward_with_softmax()

        dvalues = L.d_inputs
        ld3.backward(dvalues)
        
        dvalues = ld3.d_inputs
        activation2.backward(dvalues)
        dvalues = activation2.d_inputs
        ld2.backward(dvalues)
        
        dvalues = ld2.d_inputs
        activation1.backward(dvalues)
        dvalues = activation1.d_inputs
        ld1.backward(dvalues)
        

        opt.OptimizerGD(ld3)
        opt.OptimizerGD(ld2)
        opt.OptimizerGD(ld1)

        epoch_loss += L.loss
        epoch_acc += acc
        num_batches += 1


    epoch_avg_loss = epoch_loss / num_batches
    epoch_avg_acc = epoch_acc/num_batches

    #val
    
    ld1.forward(X_val)
    activation1.forward(ld1.output)

    ld2.forward(activation1.output)
    activation2.forward(ld2.output)

    ld3.forward(activation2.output)
    activation3.forward(ld3.output)

    L.forward(activation3.output,y_val_oh)

    val_loss = L.loss

    val_acc = accuracy(activation3.output, y_val_oh)
    
    model = Model([ld1, activation1, ld2, activation2, ld3, activation3])

    if val_loss >= lowest_Loss:
        pat += 1
        if pat >= patience :
            break
    else:
        pat = 0
        lowest_Loss = val_loss
        model.save('best_model.pkl')
        print('best model saved')

    

    print(f"Epoch {epoch+1}: train_acc: {epoch_avg_acc:.4f}, train_loss = {epoch_avg_loss:.4f}")
    print(f"Epoch {epoch+1}: val_acc: {val_acc:.4f}, val_loss = {val_loss:.4f}")



