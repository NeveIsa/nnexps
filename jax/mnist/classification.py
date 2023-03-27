import numpy as np
from jax import grad,jit
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax
from fire import Fire
from tqdm import tqdm
import plotext as plt
from helpers import gettesttrain

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

    Xtrain, ytrain, Xtest, ytest = gettesttrain() 

    lr = [lr]*(iters//3) + [10*lr]*(iters//3) + [100*lr]*(iters//3)
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
