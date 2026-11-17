from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = dict[str, jax.Array]
# ((length lo, hi), (std lo, hi)), hashable so jit can treat it as static
Bounds = tuple[tuple[float, float], tuple[float, float]]
JITTER = 1e-9


def squash(raw: jax.Array, lo: float, hi: float) -> jax.Array:
    return lo * (hi / lo) ** jax.nn.sigmoid(raw)


def unsquash(value: float, lo: float, hi: float) -> float:
    t = np.log(value / lo) / np.log(hi / lo)
    return float(np.log(t / (1 - t)))


def constrain(raw: Params, bounds: Bounds) -> Params:
    (l_lo, l_hi), (s_lo, s_hi) = bounds
    return {
        "length": squash(raw["length"], l_lo, l_hi),
        "signal": squash(raw["signal"], s_lo, s_hi),
        "noise": squash(raw["noise"], s_lo, s_hi),
    }


def init_raw(init: dict[str, float], bounds: Bounds, dim: int) -> Params:
    (l_lo, l_hi), (s_lo, s_hi) = bounds
    length = unsquash(init["length"], l_lo, l_hi)
    return {
        "length": jnp.full(dim, length, "f8"),
        "signal": jnp.asarray(unsquash(init["signal"], s_lo, s_hi), "f8"),
        "noise": jnp.asarray(unsquash(init["noise"], s_lo, s_hi), "f8"),
    }


def kernel(p: Params, a: jax.Array, b: jax.Array) -> jax.Array:
    d = (a[:, None] - b[None]) / p["length"]
    return p["signal"] ** 2 * jnp.exp(-0.5 * jnp.sum(d**2, -1))


def gram(p: Params, x: jax.Array) -> jax.Array:
    noise = p["noise"] ** 2 + JITTER
    return kernel(p, x, x) + noise * jnp.eye(len(x))


def neg_mll(
    raw: Params, bounds: Bounds, x: jax.Array, y: jax.Array
) -> jax.Array:
    chol = jnp.linalg.cholesky(gram(constrain(raw, bounds), x))
    alpha = jax.scipy.linalg.cho_solve((chol, True), y)
    log_det = jnp.sum(jnp.log(jnp.diag(chol)))
    return 0.5 * y @ alpha + log_det + 0.5 * len(y) * jnp.log(2 * jnp.pi)


@partial(jax.jit, static_argnames=("bounds", "steps"))
def fit(
    raw: Params,
    x: jax.Array,
    y: jax.Array,
    bounds: Bounds,
    steps: int,
    lr: float,
) -> Params:
    opt = optax.adam(lr)

    def body(
        carry: tuple[Params, optax.OptState], _: None
    ) -> tuple[tuple[Params, optax.OptState], None]:
        r, state = carry
        g = jax.grad(neg_mll)(r, bounds, x, y)
        updates, state = opt.update(g, state)
        return (eqx.apply_updates(r, updates), state), None

    (raw, _), _ = jax.lax.scan(body, (raw, opt.init(raw)), None, length=steps)
    return raw


def grad_cross(p: Params, x: jax.Array) -> jax.Array:
    # d k(x0, x_i) / d x0 at x0 = 0, shape (dim, n)
    k0 = kernel(p, jnp.zeros((1, x.shape[1])), x)[0]
    return (x / p["length"] ** 2).T * k0


def grad_cov(p: Params, x: jax.Array) -> jax.Array:
    # posterior covariance of grad f(0) given inputs x (independent of y)
    g = grad_cross(p, x)
    prior = jnp.diag(p["signal"] ** 2 / p["length"] ** 2)
    return prior - g @ jnp.linalg.solve(gram(p, x), g.T)


@jax.jit
def grad_mean(p: Params, x: jax.Array, y: jax.Array) -> jax.Array:
    return grad_cross(p, x) @ jnp.linalg.solve(gram(p, x), y)


@partial(jax.jit, static_argnames=("starts", "steps"))
def best_query(
    p: Params,
    x: jax.Array,
    key: jax.Array,
    radius: float,
    starts: int,
    steps: int,
) -> tuple[jax.Array, jax.Array]:
    # GIBO acquisition: the point in |z| <= radius that minimizes the trace
    # of the gradient covariance, projected Adam from random starts
    def trace(z: jax.Array) -> jax.Array:
        return jnp.trace(grad_cov(p, jnp.concatenate([x, z[None]])))

    def project(z: jax.Array) -> jax.Array:
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
        return z / jnp.maximum(1.0, norm / radius)

    dim = x.shape[1]
    z = project(jax.random.normal(key, (starts, dim), "f8") / jnp.sqrt(dim))
    opt = optax.adam(0.05 * radius)
    grad = jax.vmap(jax.grad(trace))

    def body(
        carry: tuple[jax.Array, optax.OptState], _: None
    ) -> tuple[tuple[jax.Array, optax.OptState], None]:
        z, state = carry
        updates, state = opt.update(grad(z), state)
        return (project(eqx.apply_updates(z, updates)), state), None

    (z, _), _ = jax.lax.scan(body, (z, opt.init(z)), None, length=steps)
    values = jax.vmap(trace)(z)
    best = jnp.argmin(values)
    return z[best], values[best]


def acquire(
    p: Params, x: jax.Array, n: int, key: jax.Array, cfg: dict[str, float]
) -> jax.Array:
    # greedy batch: each new point joins the inputs before the next is chosen
    picked = []
    for i in range(n):
        z, _ = best_query(
            p,
            x,
            jax.random.fold_in(key, i),
            cfg["radius"],
            int(cfg["starts"]),
            int(cfg["acq_steps"]),
        )
        picked.append(z)
        x = jnp.concatenate([x, z[None]])
    return jnp.stack(picked)
