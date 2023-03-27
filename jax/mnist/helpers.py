from sklearn.datasets import load_digits
import numpy as np

def gettesttrain():
    digits = load_digits()
    X = digits['images'].reshape(-1,64)
    y = digits['target']
    idx = np.where((y==1)|(y==0))
    y = y[idx]
    # y[y==0] = -1
    X = X[idx]
    N = len(y)
    X = (X-X.mean())/X.std()**2

    idtrain = np.random.choice(range(N), 300,replace=False)
    idtest = list(set(range(N)) - set(idtrain))
    Xtrain,ytrain = X[idtrain], y[idtrain]
    Xtest,ytest = X[idtest], y[idtest]

    return Xtrain,ytrain,Xtest,ytest
