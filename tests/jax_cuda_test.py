import time

import jax
import jax.numpy as jnp
from jax import jit, random


def main():
    print("JAX version:", jax.__version__)
    print("Default backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("Local devices:", jax.local_devices())

    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError("No JAX GPU devices found.")

    dev = gpu_devices[0]
    print("Using device:", dev)

    key = random.PRNGKey(0)

    # Put data explicitly on the GPU
    a = random.normal(key, (4096, 4096), dtype=jnp.float32)
    b = random.normal(random.PRNGKey(1), (4096, 4096), dtype=jnp.float32)
    a = jax.device_put(a, dev)
    b = jax.device_put(b, dev)

    print("a device:", a.device)
    print("b device:", b.device)

    @jit
    def test_fn(x, y):
        z = x @ y
        z = jnp.tanh(z)
        return jnp.sum(z)

    # First call: compile + run
    t0 = time.time()
    out1 = test_fn(a, b)
    out1.block_until_ready()
    t1 = time.time()

    # Second call: run only
    out2 = test_fn(a, b)
    out2.block_until_ready()
    t2 = time.time()

    print("Output 1:", out1)
    print("Output 2:", out2)
    print(f"First call (compile + run): {t1 - t0:.3f} s")
    print(f"Second call (run only):     {t2 - t1:.3f} s")

    # Sanity-check device placement of result
    print("Result device:", out2.device)

    # Small extra test: elementwise kernel
    @jit
    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    x = random.uniform(random.PRNGKey(2), (1_000_000,), dtype=jnp.float32)
    x = jax.device_put(x, dev)

    t3 = time.time()
    y = selu(x)
    y.block_until_ready()
    t4 = time.time()

    print("SELU result device:", y.device)
    print(f"SELU run time: {t4 - t3:.3f} s")


if __name__ == "__main__":
    main()
