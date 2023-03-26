import numpy as np
from jax import grad,jit
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax
from sklearn.datasets import load_digits
from fire import Fire
from tqdm import tqdm
import plotext as plt

def nnet(params, X, y):
    W = params[0]
    b = params[1]
    N = len(y)
    yhat = jax.nn.sigmoid(X@W + b)
    # plt.hist(yhat)
    # plt.show()
    diff = yhat.reshape(-1) - y
    mse = (diff**2).mean()
    return mse

def train(X,y,params,lr):
    df = grad(nnet)
    pbar = tqdm(lr)
    for lrate in pbar:
        g = df(params,X,y)
        params = tree_map(lambda x,y: x-lrate*y, params, g)
        pbar.set_postfix({'loss':nnet(params,X,y)})
    return params
    
def main(lr=0.01,iters=100):
    fanin,fanout = 64,1
    W = np.random.normal(0,np.sqrt(2/(fanin)),(fanin,fanout))
    # W = np.random.rand(fanin,fanout)*0.1
    b = np.zeros(fanout)

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
    

    lr = [lr]*100 + [10*lr]*100 + [100*lr]*100
    params = train(Xtrain,ytrain,[W,b], lr=lr)
    print (nnet(params,Xtest,ytest))
    # print(Xtest[0].tolist())
    for a,b in zip(Xtest,ytest): 
        plt.matrix_plot(a.reshape(8,8).tolist())
        plt.title(str(b))
        plt.show()
        input()
    
        
if __name__=="__main__":
    Fire(main)
