import tensorly as tl
import jax
from jax import grad,jit
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from tqdm import tqdm
from fire import Fire

@jit
def lossfn(factors,tensor,mask):
    a,b,c = factors
    tensorhat = jnp.einsum('ir,jr,kr->ijk', a,b,c)
    diff = ((tensor - tensorhat)*mask)**2
    mse = diff.mean()
    return mse

def train(factors, tensor, mask, lr):
    dl = jit(grad(lossfn))
    pbar = tqdm(lr)
    for lrate in pbar:
        gfactors = dl(factors, tensor, mask)
        factors = tree_map(lambda x,y: x- lrate*y, factors, gfactors)
        pbar.set_postfix({'loss':lossfn(factors, tensor, mask)})
    return factors

def main(rank=1,gdrank=1,mp=50,lr=0.1,iters=100):
    lr = [lr]*iters
    RANK=rank
    MP = mp
    cpt = tl.random.random_cp((3,4,5), rank=RANK)
    w,(a,b,c) = cpt 

    # makes weights on
    w = np.ones(RANK)
    cpt[0] = 1

    t = cpt.to_tensor()
    t = t*100
    mask = (np.random.rand(*t.shape)>(MP/100))*1

    GDRANK = gdrank
    ia,ib,ic = map(lambda x:1*np.random.rand(x.shape[0],GDRANK),[a,b,c])
    fa,fb,fc = train([ia,ib,ic],t,mask,lr)

    # testloss = (mask)*(  tl.cp_to_tensor((np.ones(GDRANK),(fa,fb,fc))) - t  )
    testloss = lossfn([fa,fb,fc],t,1-mask)
    print(testloss)
    # print(lossfn([fa,fb,fc], t, mask))

if __name__ == "__main__":
    Fire(main)
