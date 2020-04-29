# Submitted by Amin Shojaeighadikolaei
# 04/28/2020

##################################################################### Libraries

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K

#################################################################### Functions
def One_hot(X):   
    onehot_encoder = OneHotEncoder(sparse=False)
    temp = X.reshape(len(X), 1)
    onehots = onehot_encoder.fit_transform(temp)
    return(onehots)


def normalize(data):
    return ((data - data.mean()) / np.sqrt(np.var(data)))

def sigmoid(data):
    return 1 / (1 + np.exp(-data))

################################################################ Create samples
input_size = 8
output_size = 6
hidden_layers = [4, 3]
nSamples = 100000

# randomly pick joint distribution, normalize
Pxy = np.random.random([output_size, input_size])
Pxy = Pxy / np.sum(np.sum(Pxy,axis=0),axis=0)
#  # compute marginals
Px = np.sum(Pxy, axis = 0)
Py = np.sum(Pxy, axis = 1)

X=np.random.choice(range(input_size),nSamples,p=Px)
Y=np.random.choice(range(output_size),nSamples,p=Py)

x_train = One_hot(X)
y_train = One_hot(Y)
################################################################## Create model
model = Sequential()
model.add(Dense(hidden_layers[0], activation='sigmoid', input_dim=input_size))
model.add(Dense(hidden_layers[1], activation='softmax', input_dim=hidden_layers[0]))    
model.add(Dense(output_size, activation='softmax', input_dim=hidden_layers[1]))

sgd = SGD(lr=4, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=.01) 
model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

######################################################Create Information matrix
B = (Pxy- Px.reshape(1, -1)*Py.reshape(-1, 1))/ np.sqrt(Px.reshape(1, -1)) * np.sqrt(Py.reshape(-1, 1))
phi_y, phi, phi_x = np.linalg.svd(B)

t = K.function([model.layers[0].input], [model.layers[0].output])([np.eye(input_size)])[0]
t_mean = np.matmul(Px.reshape(1, -1), t)
print("Shape of the first layer output: ", t.shape)

Phi1 = t*(np.sqrt(Px).reshape(-1,1))  # information matrix input

model.fit(x_train, y_train, verbose=0, batch_size=5000, epochs=200)
weights = model.get_weights()

################################################################ Theoretical calculation
v = model.get_weights()[4].T
Phi2 = v*(np.sqrt(Py).reshape(-1,1))  # information matrix output
b = model.get_weights()[5]
b_hat = b - np.dot(b, Py)

s = K.function([model.layers[0].input], [model.layers[1].output])([np.eye(input_size)])[0]
s_mean = np.matmul(Px.reshape(1, -1), s)


b_theory = np.log(Py) - np.matmul(v, s_mean.T).reshape(-1) 

c = model.get_weights()[3]  
c = normalize(c)
w = model.get_weights()[2]
c_theory = np.log(P_hidden) - np.matmul(w, t_mean.T).reshape(-1) 



