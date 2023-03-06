import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import timeit

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.PRNGKey(0)
x = random.uniform(key, (100000,))
#x = np.random.random((10000,))

print(selu(x).block_until_ready())
#timeit.timeit("selu(x)")

selu_jit = jit(selu)
print(selu_jit(x).block_until_ready())
#timeit.timeit("selu_jit(x)")
