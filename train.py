"""Train and evaluate the four learned controllers for the donut drift.

    quad      quadratic V = z'Pz (P learned),           NN policy
    nonquad   V = ||phi(z) - phi(0)||^2 + eps ||z||^2,  NN policy
    lqr_nn    quadratic V (P starts at Riccati),        LQR + NN residual policy
    dpc       no V: LQR + NN residual policy trained on the LQR cost through
              3 s rollouts (differentiable predictive control baseline)

The untrained 9-state LQR is evaluated alongside as a reference.

    python train.py                                # all four, seed 0
    python train.py --cases quad lqr_nn --seed 1   # a subset, another seed
    python train.py --steps 300 --quick            # fast smoke test

Writes <out>/comparison.md, <out>/comparison.json and <out>/<case>/policy.npz.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_continuous_are

import model as M

NX = 9
CASES = {"quad": ("quadratic", False), "nonquad": ("nonquad", False),   # (V form, LQR prior)
         "lqr_nn": ("quadratic", True), "dpc": ("quadratic", True)}
TITLE = {"lqr": "9-state LQR (untrained)", "quad": "Quadratic V", "nonquad": "Neural V",
         "lqr_nn": "LQR + NN, quadratic V", "dpc": "DPC baseline"}

# ---- hyperparameters ---------------------------------------------------------
BOX = np.array([0.15, math.radians(20), math.radians(25), 2.0, 0.7,     # half-widths s of the
                math.radians(20), 45.0, math.radians(20), 45.0])         # training box D
W_STATE = np.array([2.0, 1.0, 12.0, 3.0, 25.0, 0.5, 0.5, 1.0, 1.0])      # LQR priorities
W_RATE = np.array([250.0, 4.0])
LAM = dict(area=0.03, dec=3.0, bnd=100.0, lin=1.0, gain=1e-2)           # Lyapunov loss weights
MARGIN, BND_MARGIN, AREA_TAU, BALL = 0.05, 0.1, 0.1, 0.02
EPS_V, EPS_P = 0.01, 1e-3
BATCH, N_GAIN, CEX_POOL, BUFFER_B, FACE_STEPS, REFRESH = 8192, 1024, 60000, 512, 20, 100
HIDDEN_V, HIDDEN_PI, N_FEAT, PI_INIT = 32, 64, 16, 0.1
LR = 1e-3
DPC = dict(horizon0=0.5, ramp=1 / 3, init=0.5, batch=512, clip=1.0, cap=10.0, abort=3.0, vmin=0.3)


# ---- setup: equilibrium, plant, LQR ------------------------------------------
def setup(args):
    m, lim = M.Vehicle(), M.Limits()
    eq = M.solve_equilibrium(m, args.target_speed, args.target_radius, lim=lim)
    fold = M.DELAY_TICKS * M.DT                     # loop delay folded into the training lags
    f = M.make_plant(m, eq.curvature, M.SERVO_TAU + fold, M.DRIVE_TAU + fold)
    xeq = jnp.asarray(eq.x9())
    A = np.asarray(jax.jacfwd(lambda x: f(x, jnp.zeros(2)))(xeq))
    B = np.asarray(jax.jacfwd(lambda u: f(xeq, u))(jnp.zeros(2)))
    maxe = np.r_[M.MAX_ERROR, M.MAX_ERROR[5:7]]
    Q = np.diag(W_STATE / maxe ** 2)
    R = np.diag(W_RATE / M.MAX_RATE ** 2)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    alpha_lqr = float(np.min(np.real(np.linalg.eigvals(np.linalg.solve(P, Q + K.T @ R @ K)))))
    s = BOX.copy()
    Pn = np.diag(s) @ P @ np.diag(s)
    return dict(m=m, lim=lim, eq=eq, f=f, xeq=xeq, s=jnp.asarray(s), Q=Q, R=R, P=P, K=K,
                Kj=jnp.asarray(K), Pn=jnp.asarray(Pn / (np.trace(Pn) / NX)), alpha=0.5 * alpha_lqr,
                umax=jnp.asarray([lim.ddelta_max, lim.domega_max]))


# ---- networks, V and policy --------------------------------------------------
def mlp_init(key, sizes, last_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        std = math.sqrt(1.0 / n_in) * (last_scale if i == len(sizes) - 2 else 1.0)
        layers.append((jax.random.normal(k, (n_in, n_out)) * std, jnp.zeros(n_out)))
    return layers


def mlp(layers, x):
    for W, b in layers[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = layers[-1]
    return x @ W + b


def init_params(key, S, cfg):
    v_form, use_lqr = cfg
    w, Qe = np.linalg.eigh((np.asarray(S["Pn"]) if use_lqr else np.eye(NX)) - EPS_P * np.eye(NX))
    k1, k2 = jax.random.split(key)
    return {"L": jnp.asarray(Qe @ np.diag(np.sqrt(np.maximum(w, 1e-9)))),
            "phi": mlp_init(k1, [NX, HIDDEN_V, HIDDEN_V, N_FEAT], 1.0 if v_form == "nonquad" else 0.1),
            "g": mlp_init(k2, [NX, HIDDEN_PI, HIDDEN_PI, 2], 0.0 if use_lqr else PI_INIT)}


def build_fns(S, cfg):
    v_form, use_lqr = cfg
    f, xeq, s, Kj, umax = S["f"], S["xeq"], S["s"], S["Kj"], S["umax"]
    z0, I = jnp.zeros(NX), jnp.eye(NX)

    def Pmat(p):
        return EPS_V * I if v_form == "nonquad" else p["L"] @ p["L"].T + EPS_P * I

    def V(p, z):
        v = z @ Pmat(p) @ z
        if v_form == "nonquad":
            d = mlp(p["phi"], z) - mlp(p["phi"], z0)
            v = v + d @ d
        return v

    def policy(p, z):
        corr = mlp(p["g"], z) - mlp(p["g"], z0)
        if use_lqr:
            return umax * jnp.tanh((-Kj @ (z * s) + umax * corr) / umax)
        return umax * jnp.tanh(corr)

    def vdot(p, z):
        return jax.jvp(lambda zz: V(p, zz), (z,), (f(xeq + z * s, policy(p, z)) / s,))

    def A_cl(p):
        return jax.jacfwd(lambda z: f(xeq + z * s, policy(p, z)) / s)(z0)

    def lin_rate(p):
        """Largest rho with A'H + HA <= -rho H at x*, H = Hessian(V)/2."""
        H = 0.5 * jax.hessian(lambda z: V(p, z))(z0)
        Li = jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(H), I, lower=True)
        A = A_cl(p)
        Mm = Li @ (A.T @ H + H @ A) @ Li.T
        return -jnp.max(jnp.linalg.eigvalsh(0.5 * (Mm + Mm.T)))

    jpol = jax.vmap(jax.jacfwd(lambda p, z: policy(p, z) / umax, argnums=1), in_axes=(None, 0))
    return dict(V=V, policy=policy, Pmat=Pmat, lin_rate=lin_rate, A_cl=A_cl,
                Vb=jax.vmap(V, in_axes=(None, 0)), vdotb=jax.vmap(vdot, in_axes=(None, 0)),
                gain=lambda p, Z: jnp.mean(jnp.sum(jpol(p, Z) ** 2, axis=(1, 2))))


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": z, "t": 0}


def adam_step(params, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], g)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


def face_descent(V, params, Zf, steps, h0, h1):
    """Move box-face points toward lower V along their face (sign-gradient steps h0 -> h1)."""
    free = (jnp.abs(Zf) < 1.0 - 1e-9).astype(Zf.dtype)
    grad = jax.vmap(jax.grad(lambda z: V(params, z)))

    def body(i, Z):
        h = h0 * (h1 / h0) ** (i / max(steps - 1, 1))
        return jnp.clip(Z - h * jnp.sign(grad(Z)) * free, -1.0, 1.0)

    Z = jax.lax.fori_loop(0, steps, body, Zf)
    return Z, jax.vmap(lambda z: V(params, z))(Z)


# ---- Lyapunov training (quad, nonquad, lqr_nn) --------------------------------
def train_lyapunov(S, args, cfg, tag):
    F, alpha = build_fns(S, cfg), S["alpha"]
    rng = np.random.default_rng(args.seed)

    def faces(k):
        Zb = rng.uniform(-1, 1, (k, NX))
        Zb[np.arange(k), rng.integers(0, NX, k)] = rng.choice([-1.0, 1.0], k)
        return Zb

    def loss_fn(p, Z, Zb, Zm):
        v, vd = F["vdotb"](p, Z)
        w = (jnp.max(jnp.abs(Z), axis=1) > BALL).astype(Z.dtype)
        dec = jnp.sum(jax.nn.relu(vd / (v + 1e-12) + alpha + MARGIN) * w) / jnp.maximum(jnp.sum(w), 1.0)
        area = jnp.mean(jax.nn.sigmoid((1.0 - v) / AREA_TAU))
        bnd = (jnp.mean(jax.nn.relu(1.0 + BND_MARGIN - F["Vb"](p, Zb)))
               + jnp.mean(jax.nn.relu(1.0 + BND_MARGIN - F["Vb"](p, Zm))))
        rho = F["lin_rate"](p)
        gain = F["gain"](p, Z[:N_GAIN])
        loss = (-LAM["area"] * jnp.log(area + 1e-12) + LAM["dec"] * dec + LAM["bnd"] * bnd
                + LAM["lin"] * jax.nn.relu(alpha + MARGIN - rho) + LAM["gain"] * gain)
        return loss, (dec, bnd, rho)

    @jax.jit
    def step(p, opt, Z, Zb, Zm, lr):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, Z, Zb, Zm)
        p, opt = adam_step(p, g, opt, lr)
        return p, opt, l, aux

    Vj = jax.jit(F["Vb"])
    refine = jax.jit(lambda p, Z: face_descent(F["V"], p, Z, FACE_STEPS, 0.05, 0.005))

    def mine(p):        # the lowest-V face points, refined along their face
        pool = faces(CEX_POOL)
        vp = np.asarray(Vj(p, jnp.asarray(pool)))
        return np.asarray(refine(p, jnp.asarray(pool[np.argsort(vp)[:BUFFER_B]]))[0])

    params = init_params(jax.random.PRNGKey(args.seed), S, cfg)
    opt = adam_init(params)
    mined = mine(params)
    Ztrack = jnp.asarray(np.random.default_rng(args.seed + 999).uniform(-1, 1, (50000, NX)))
    t0 = time.time()
    for it in range(args.steps + 1):
        lr = LR * (0.1 if it > 0.85 * args.steps else 1.0)
        Z, Zb = rng.uniform(-1, 1, (BATCH, NX)), faces(BATCH // 4)
        params, opt, l, (dec, bnd, rho) = step(params, opt, jnp.asarray(Z), jnp.asarray(Zb),
                                               jnp.asarray(mined), lr)
        if it % REFRESH == 0:
            mined = mine(params)
        if it % args.log_every == 0:
            vol = float(np.mean(np.asarray(Vj(params, Ztrack)) <= 1.0))
            print(f"  [{tag}] step {it:5d}  loss {float(l):+.3e}  decrease {float(dec):.2e}  "
                  f"boundary {float(bnd):.2e}  volume {100 * vol:.3f}%  rho {float(rho):+.2f}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    return params, time.time() - t0


# ---- DPC training (dpc) -----------------------------------------------------
def train_dpc(S, args, cfg, tag):
    """Minimize J/J0 + w_g * gain, J = E sum_k min(z'Q~z + u'Ru, cap) dt over rollouts of
    the training model, Q~ = SQS. No terminal cost. J0: the initial LQR policy's cost."""
    F, f, xeq, s, lim = build_fns(S, cfg), S["f"], S["xeq"], S["s"], S["lim"]
    dt = M.DT
    h = dt / 2
    lo = jnp.array([-jnp.inf] * 5 + [-lim.delta_max, lim.omega_min] * 2)
    hi = jnp.array([jnp.inf] * 5 + [lim.delta_max, lim.omega_max] * 2)
    Qn = jnp.asarray(np.diag(np.asarray(s)) @ S["Q"] @ np.diag(np.asarray(s)))
    Rm = jnp.asarray(S["R"])
    N = int(round(args.dpc_horizon_s / dt))
    N0 = max(1, int(round(DPC["horizon0"] / dt)))
    ramp = max(1, int(DPC["ramp"] * args.steps))
    rng = np.random.default_rng(args.seed)
    bound = 1.5 * DPC["abort"]
    zV_min = float((DPC["vmin"] - float(xeq[4])) / float(s[4]))
    Zc = rng.uniform(-1, 1, (4096, NX))
    cap = float(DPC["cap"] * np.mean(np.einsum("ij,jk,ik->i", Zc, np.asarray(Qn), Zc)) * dt)

    def rk4(x, u):
        k1 = f(x, u); k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u); k4 = f(x + h * k3, u)
        return jnp.clip(x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, lo, hi)

    def traj_cost(p, z0, n_cur):
        def tick(carry, k):
            z, J, alive = carry
            u = F["policy"](p, z)
            c = (z @ Qn @ z + u @ Rm @ u) * dt
            w = (k < n_cur).astype(z.dtype)                    # horizon curriculum
            J = J + w * (alive * jnp.minimum(c, cap) + (1.0 - alive) * cap)
            x = rk4(rk4(xeq + z * s, u), u)
            zn = (x - xeq) / s
            ok = jnp.all(jnp.isfinite(zn)) & (jnp.max(jnp.abs(zn)) < DPC["abort"]) & (x[4] > DPC["vmin"])
            alive = alive * ok.astype(z.dtype)
            zn = jnp.where(ok, zn, jnp.zeros_like(zn))        # a dead rollout parks at x* and pays the cap
            zn = jnp.clip(zn, -bound, bound)
            zn = zn.at[4].set(jnp.maximum(zn[4], zV_min))
            return (zn, J, alive), None
        (_, J, alive), _ = jax.lax.scan(tick, (z0, jnp.asarray(0.0), jnp.asarray(1.0)), jnp.arange(N))
        return J, alive

    batch_cost = jax.vmap(traj_cost, in_axes=(None, 0, None))

    def loss_fn(p, Z0, Zg, n_cur, J0):
        J, alive = batch_cost(p, Z0, n_cur)
        gain = F["gain"](p, Zg)
        return jnp.mean(J) / J0 + LAM["gain"] * gain, (jnp.mean(J), jnp.mean(alive))

    @jax.jit
    def step(p, opt, Z0, Zg, n_cur, J0, lr):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, Z0, Zg, n_cur, J0)
        g = jax.tree_util.tree_map(jnp.nan_to_num, g)
        gn = jnp.sqrt(sum(jnp.sum(t ** 2) for t in jax.tree_util.tree_leaves(g)))
        g = jax.tree_util.tree_map(lambda t: t * jnp.minimum(1.0, DPC["clip"] / (gn + 1e-12)), g)
        p, opt = adam_step(p, g, opt, lr)
        return p, opt, l, aux

    params = init_params(jax.random.PRNGKey(args.seed), S, cfg)
    opt = adam_init(params)
    Zfix = jnp.asarray(np.random.default_rng(args.seed + 4242).uniform(-DPC["init"], DPC["init"], (DPC["batch"], NX)))
    J0 = float(jnp.mean(jax.jit(batch_cost)(params, Zfix, N)[0]))
    t0 = time.time()
    for it in range(args.steps + 1):
        lr = LR * (0.1 if it > 0.85 * args.steps else 1.0)
        n_cur = int(round(N0 + (N - N0) * min(1.0, it / ramp)))
        Z0 = jnp.asarray(rng.uniform(-DPC["init"], DPC["init"], (DPC["batch"], NX)))
        Zg = jnp.asarray(rng.uniform(-1, 1, (N_GAIN, NX)))
        params, opt, l, (Jm, alive) = step(params, opt, Z0, Zg, n_cur, J0, lr)
        if it % args.log_every == 0:
            print(f"  [{tag}] step {it:5d}  loss {float(l):.4f}  J/J0 {float(Jm) / J0:.4f}  "
                  f"horizon {n_cur * dt:.2f} s  alive {100 * float(alive):5.1f}%  [{time.time() - t0:.0f} s]",
                  flush=True)
    return params, time.time() - t0


# ---- evaluation ----------------------------------------------------------------
def sampled_checks(F, params, alpha, n, seed):
    """Volume of {V <= 1}, and samples with Vdot > 0 (|Vdot/V| > 1e-3) on the box and inside the set."""
    rng = np.random.default_rng(seed)
    fn = jax.jit(F["vdotb"])
    n_set = n_in = n_box = pos_box = pos_in = 0
    for i in range(0, n, 100_000):
        Z = rng.uniform(-1, 1, (min(100_000, n - i), NX))
        v, vd = (np.asarray(t) for t in fn(params, jnp.asarray(Z)))
        keep = np.max(np.abs(Z), axis=1) > BALL
        pos = vd[keep] / (v[keep] + 1e-12) > 1e-3
        inside = v[keep] <= 1.0
        n_set += int(np.sum(v <= 1.0))
        n_box += int(np.sum(keep)); n_in += int(np.sum(inside))
        pos_box += int(np.sum(pos)); pos_in += int(np.sum(pos & inside))
    eig = np.linalg.eigvals(np.asarray(F["A_cl"](params)))
    return dict(volume=n_set / n, n_in=n_in, n_box=n_box, pos_in=pos_in, pos_box=pos_box,
                rho=float(F["lin_rate"](params)), slowest_eig=float(np.max(eig.real)))


def face_min(F, params, seed):
    """Lowest V on the box faces (>= 1 means {V <= 1} lies inside the box)."""
    rng = np.random.default_rng(seed)
    Zb = rng.uniform(-1, 1, (250000, NX))
    Zb[np.arange(len(Zb)), rng.integers(0, NX, len(Zb))] = rng.choice([-1.0, 1.0], len(Zb))
    vb = np.asarray(jax.jit(F["Vb"])(params, jnp.asarray(Zb)))
    _, vf = jax.jit(lambda p, Z: face_descent(F["V"], p, Z, 400, 0.05, 1e-4))(
        params, jnp.asarray(Zb[np.argsort(vb)[:512]]))
    return float(min(vb.min(), float(np.min(np.asarray(vf)))))


def policy_gain(S, F, params, n=400, seed=5):
    """Spectral norm of d(u/umax)/dx: at x* and 95th percentile over the box."""
    s, umax = np.asarray(S["s"]), np.asarray(S["umax"])
    J = jax.jit(jax.vmap(jax.jacfwd(F["policy"], argnums=1), in_axes=(None, 0)))
    Z = np.random.default_rng(seed).uniform(-1, 1, (n, NX))
    Z[0] = 0.0
    norms = np.linalg.norm(np.asarray(J(params, jnp.asarray(Z))) / s[None, None, :] / umax[None, :, None],
                           ord=2, axis=(1, 2))
    return float(norms[0]), float(np.percentile(norms, 95))


def make_rollout(S, F, delay_ticks=M.DELAY_TICKS, vehicle=None, training_model=False):
    """Closed loop at 100 Hz: real lags and transport delay (or the training model)."""
    lim, xeq, s, policy = S["lim"], S["xeq"], S["s"], F["policy"]
    if training_model:
        f, nd = S["f"], 0
    else:
        f, nd = M.make_plant(vehicle or S["m"], S["eq"].curvature, M.SERVO_TAU, M.DRIVE_TAU), delay_ticks
    h = M.DT / 2
    lo = jnp.array([-jnp.inf] * 5 + [-lim.delta_max, lim.omega_min] * 2)
    hi = jnp.array([jnp.inf] * 5 + [lim.delta_max, lim.omega_max] * 2)

    def rk4(x, u):
        k1 = f(x, u); k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u); k4 = f(x + h * k3, u)
        return jnp.clip(x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, lo, hi)

    def run(params, x0, n_ticks):
        def tick(carry, _):
            x, buf = carry
            u_cmd = policy(params, (x - xeq) / s)
            if nd > 0:
                u, buf = buf[0], jnp.concatenate([buf[1:], u_cmd[None]])
            else:
                u = u_cmd
            x = rk4(rk4(x, u), u)
            x = jnp.where(jnp.all(jnp.isfinite(x)), x, jnp.full_like(x, jnp.nan))
            return (x, buf), x[:7]
        return jax.lax.scan(tick, (x0, jnp.zeros((max(nd, 1), 2))), None, length=n_ticks)[1]

    return jax.jit(jax.vmap(run, in_axes=(None, 0, None)), static_argnums=2)


TOL_EQ = np.array([0.03, math.radians(4.0), math.radians(3.0), 0.15, 0.10, math.radians(4.0), 6.0])
TOL_STEADY = np.array([0.02, math.radians(2.0), math.radians(1.5), 0.08, 0.05, math.radians(1.5), 3.0])


def converged(tr, x7, n):
    """Within TOL_EQ of x* over the last n ticks."""
    e = np.abs(np.nan_to_num(tr[:, -n:, :], nan=1e9) - x7)
    return np.all(np.isfinite(tr), axis=(1, 2)) & np.all(e < TOL_EQ, axis=(1, 2))


def settled(tr, x7, R, n):
    """Settled on some steady donut near the circle over the last n ticks."""
    w = np.nan_to_num(tr[:, -n:, :], nan=1e9)
    steady = np.all(np.abs(w - w[:, -1:, :]) < TOL_STEADY, axis=(1, 2))
    last = w[:, -1, :]
    ok = ((np.abs(last[:, 2] - x7[2]) < math.radians(30.0)) & (last[:, 3] * np.sign(x7[3]) > 0.5)
          & (np.abs(last[:, 0]) < 0.5 * R))
    return np.all(np.isfinite(tr), axis=(1, 2)) & steady & ok


def evaluate(S, F, params, X7s, n_robust):
    x7 = np.asarray(S["xeq"])[:7]
    n_ticks, settle = int(8.0 / M.DT), int(1.0 / M.DT)
    full = lambda X: jnp.asarray(np.c_[X, X[:, 5], X[:, 6]])     # references start at the measured state
    run = make_rollout(S, F)
    out = {"box": [float(np.mean(converged(np.asarray(run(params, full(X), n_ticks)), x7, settle))) for X in X7s]}
    X0 = full(X7s[0][:n_robust])
    out["delay"] = {d: float(np.mean(settled(np.asarray(make_rollout(S, F, delay_ticks=d)(params, X0, n_ticks)),
                                             x7, S["eq"].radius, settle))) for d in (2, 4, 6, 8)}
    out["tire"] = {}
    for k in (0.90, 0.95, 1.05, 1.10):
        m2 = dataclasses.replace(S["m"], rear_high_force=S["m"].rear_high_force * k, front_D=S["m"].front_D * k)
        tr = np.asarray(make_rollout(S, F, vehicle=m2)(params, X0, int(1.5 * n_ticks)))
        out["tire"][k] = float(np.mean(settled(tr, x7, S["eq"].radius, settle)))
    # rollouts started inside {V <= 1} on the training model
    Z = np.random.default_rng(11).uniform(-1, 1, (200000, NX))
    Zin = Z[np.asarray(jax.jit(F["Vb"])(params, jnp.asarray(Z))) < 1.0][:300]
    if len(Zin):
        X = np.asarray(S["xeq"]) + Zin * np.asarray(S["s"])
        tr = np.asarray(make_rollout(S, F, training_model=True)(params, jnp.asarray(X), int(12.0 / M.DT)))
        out["inside"] = (int(np.sum(converged(tr, x7, int(0.5 / M.DT)))), len(Zin))
    else:
        out["inside"] = (0, 0)
    return out


def export(S, F, cfg, params, path):
    """policy.npz: u = umax tanh((-K (x - x*) + umax (g(z) - g(0))) / umax), z = (x - x*)/s,
    V = z' P_norm z + ||phi(z) - phi(0)||^2 (phi is zero for quadratic V)."""
    phi = list(params["phi"])
    if cfg[0] == "quadratic":
        phi[-1] = tuple(jnp.zeros_like(t) for t in phi[-1])
    flat = {f"{pre}_{w}{i}": np.asarray(t) for pre, layers in (("phi", phi), ("g", params["g"]))
            for i, (W, b) in enumerate(layers) for w, t in (("W", W), ("b", b))}
    np.savez(path, x_eq=np.asarray(S["xeq"]), s=np.asarray(S["s"]), P_norm=np.asarray(F["Pmat"](params)),
             K=S["K"] if cfg[1] else np.zeros_like(S["K"]), umax=np.asarray(S["umax"]), c=1.0, plant="pi", **flat)


# ---- main ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--target-speed", type=float, default=1.356, help="donut speed [m/s]")
    p.add_argument("--target-radius", type=float, default=0.50, help="donut radius [m]")
    p.add_argument("--dpc-horizon-s", type=float, default=3.0)
    p.add_argument("--quick", action="store_true", help="smaller evaluation samples (smoke test)")
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    n_check, n_roll, n_robust = (100_000, 200, 100) if args.quick else (1_000_000, 1000, 400)
    args.out.mkdir(parents=True, exist_ok=True)

    S = setup(args)
    eq = S["eq"]
    print(f"donut V = {eq.V} m/s, R = {eq.radius} m: beta {math.degrees(eq.beta):+.2f} deg, "
          f"delta {math.degrees(eq.delta):+.2f} deg, omega {eq.omega:.2f} rad/s; required decay "
          f"alpha = {S['alpha']:.3f} 1/s")

    R = {"lqr": {"params": init_params(jax.random.PRNGKey(0), S, ("quadratic", True)), "train_s": None}}
    for nm in args.cases:
        print(f"\nTRAINING {nm} ({TITLE[nm]})")
        trainer = train_dpc if nm == "dpc" else train_lyapunov
        params, t = trainer(S, args, CASES[nm], nm)
        R[nm] = {"params": params, "train_s": t}

    # initial states for the closed-loop tests: box x0.5 and x1.0 around x*, physical limits
    rng = np.random.default_rng(args.seed + 7)
    s7, x7 = np.asarray(S["s"])[:7], np.asarray(S["xeq"])[:7]
    lo = np.array([-np.inf, -np.inf, math.radians(-80.0), 0.2, 0.4, -S["lim"].delta_max, 0.0])
    hi = np.array([np.inf, np.inf, math.radians(-5.0), 8.0, 4.0, S["lim"].delta_max, S["lim"].omega_max])
    X7s = [np.clip(x7 + rng.uniform(-q, q, (n_roll, 7)) * s7, lo, hi) for q in (0.5, 1.0)]

    print("\nEVALUATING")
    for nm, r in R.items():
        cfg = ("quadratic", True) if nm == "lqr" else CASES[nm]
        F = build_fns(S, cfg)
        r["check"] = sampled_checks(F, r["params"], S["alpha"], n_check, args.seed + 101)
        r["face_min"] = face_min(F, r["params"], args.seed + 202)
        r["gain"] = policy_gain(S, F, r["params"])
        r["roll"] = evaluate(S, F, r["params"], X7s, n_robust)
        if nm != "lqr":
            (args.out / nm).mkdir(exist_ok=True)
            export(S, F, cfg, r["params"], args.out / nm / "policy.npz")
        print(f"  {nm} done", flush=True)

    note = " (DPC has no V of its own: V rows use the LQR's Riccati V)" if "dpc" in R else ""
    rows = [("training time [s]", lambda r: "-" if r["train_s"] is None else f"{r['train_s']:.0f}"),
            ("{V <= 1}, % of box volume", lambda r: f"{100 * r['check']['volume']:.4f}"),
            ("lowest V on the box faces (>= 1: set inside the box)", lambda r: f"{r['face_min']:.3f}"),
            ("Vdot > 0 on the box, % (count)",
             lambda r: f"{100 * r['check']['pos_box'] / r['check']['n_box']:.3f} ({r['check']['pos_box']})"),
            ("Vdot > 0 inside {V <= 1}, count / samples",
             lambda r: f"{r['check']['pos_in']} / {r['check']['n_in']}"),
            ("local decay rate rho at x* [1/s]", lambda r: f"{r['check']['rho']:.2f}"),
            ("slowest closed-loop eigenvalue [1/s]", lambda r: f"{r['check']['slowest_eig']:+.2f}"),
            ("policy gain at x* / 95th pct over box", lambda r: f"{r['gain'][0]:.2f} / {r['gain'][1]:.2f}"),
            ("rollouts from inside {V <= 1} that converge", lambda r: "%d / %d" % r["roll"]["inside"]),
            ("converged from box x0.5", lambda r: f"{100 * r['roll']['box'][0]:.0f}%"),
            ("converged from box x1.0", lambda r: f"{100 * r['roll']['box'][1]:.0f}%")]
    rows += [(f"settled, delay {d} ticks", lambda r, d=d: f"{100 * r['roll']['delay'][d]:.0f}%") for d in (2, 4, 6, 8)]
    rows += [(f"settled, tire x{k:.2f}", lambda r, k=k: f"{100 * r['roll']['tire'][k]:.0f}%")
             for k in (0.90, 0.95, 1.05, 1.10)]
    table = ("| | " + " | ".join(TITLE[k] for k in R) + " |\n|---|" + "---|" * len(R) + "\n"
             + "\n".join(f"| {lab} | " + " | ".join(fn(r) for r in R.values()) + " |" for lab, fn in rows))
    head = (f"# Donut V = {eq.V} m/s, R = {eq.radius} m; seed {args.seed}; {args.steps} steps; "
            f"alpha = {S['alpha']:.3f} 1/s{note}\n\n")
    (args.out / "comparison.md").write_text(head + table + "\n")
    (args.out / "comparison.json").write_text(json.dumps(
        {nm: {k: v for k, v in r.items() if k != "params"} for nm, r in R.items()}, indent=2, default=str))
    print("\n" + table + f"\n\nwrote {args.out}/comparison.md, comparison.json, <case>/policy.npz")


if __name__ == "__main__":
    main()
