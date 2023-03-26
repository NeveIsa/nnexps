from helpers import gettesttrain
import numpy as np
from jax import grad,jit
import jax
import plotext as plt
from jax.tree_util import tree_map
from tqdm import tqdm
from fire import Fire

@jit
def encoder(params, X):
    for p in params:
        X = X@p
        X =jax.nn.sigmoid(X)
    return X

@jit
def decoder(params,encoded):
    for p in params:
        encoded = encoded@p
        encoded = jax.nn.sigmoid(encoded)
    return encoded
    
def autoencoder(params,X):
    enc = encoder(params[0],X)
    dec = decoder(params[1], enc)
    return dec

@jit
def lossfn(params, X):
    Xhat = autoencoder(params,X)
    # assert Xhat.shape == X.shape
    loss =((Xhat - X)**2).mean()
    return loss

def train(params, X, lr):
    dlfn = jit(grad(lossfn))
    # lossfn=jit(lossfn)
    
    g = dlfn(params, X)

    pbar = tqdm(lr)
    for lrate in pbar:
        params = tree_map(lambda x,y:x-lrate*y, params,g)
        # pbar.set_postfix({'loss':lossfn(params,X)})

    return params


def plot(params,X, label):
    enc = encoder(params,X)
    x1 = enc[:,0].tolist()
    x2 = enc[:,1].tolist()
    plt.scatter(x1, x2, label=label)
    
def main(lr=0.1,iters=100):
    encparams = [np.random.rand(64,10),np.random.rand(10,5),np.random.rand(5,2) ]
    encparams = [e/np.sqrt(sum(e.shape)) for e in encparams]
    decparams = [np.random.rand(2,5),np.random.rand(5,10),np.random.rand(10,64) ]
    decparams = [d/np.sqrt(sum(d.shape)) for d in decparams]
    
    params = [encparams, decparams]
    
    Xtrain,ytrain,Xtest,ytest = gettesttrain()
    Xtrain=Xtrain*10
    Xtest=Xtest*10
    
    lr = [lr]*iters + [lr/10]*iters
    params = train(params, Xtrain, lr)

    X0 = Xtest[ytest==0]
    X1 = Xtest[ytest==1] 
    
    plot(params[0],X0, label='0')
    plot(params[0],X1, label='1')
    plt.show()
        
if __name__ == "__main__":
    Fire(main)   

