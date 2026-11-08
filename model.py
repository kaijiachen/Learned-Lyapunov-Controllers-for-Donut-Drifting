"""Vehicle model of the 1/10-scale car in a donut drift.

State (9):  x = [e, dphi, beta, r, V, delta, omega, delta_ref, omega_ref]
            path error [m], heading error [rad], sideslip [rad], yaw rate [rad/s],
            speed [m/s], steering angle [rad], rear wheel speed [rad/s], and the two
            references tracked by the steering servo and the onboard wheel-speed PI.
Input  (2): u = [d(delta_ref)/dt, d(omega_ref)/dt]

Chassis: single-track model with a Pacejka front tire and a combined-slip
friction-circle rear tire. Path: Frenet kinematics around a circle of
curvature kappa. Actuators: first-order lags from each reference to its state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

LABELS = ("e", "dphi", "beta", "r", "V", "delta", "omega", "delta_ref", "omega_ref")

# Reference scales for the LQR cost: Q = diag(w / MAX_ERROR^2), R = diag(w / MAX_RATE^2)
MAX_ERROR = np.array([0.10, math.radians(10.0), math.radians(10.0), 0.40, 0.35,
                      math.radians(12.0), 18.0])
MAX_RATE = np.array([math.radians(300.0), 120.0])

# Loop timing and actuator lags measured on the car
DT = 0.01               # control period [s]
SERVO_TAU = 0.045       # delta_ref -> delta [s]
DRIVE_TAU = 0.05        # omega_ref -> omega, onboard PI closed loop [s]
DELAY_TICKS = 2         # transport delay PC -> car [ticks]


@dataclass(frozen=True)
class Vehicle:
    mass: float = 1.9580
    a_cg: float = 0.1347
    b_cg: float = 0.1223
    izz: float = 0.0400
    wheel_radius: float = 0.0350
    # front tire: Fyf = -MF(alpha_f), Pacejka B, C, D, E
    front_B: float = 10.2585823
    front_C: float = 1.07668211
    front_D: float = 3.46461411
    front_E: float = -2.0
    # rear tire, low slip: longitudinal slope and lateral Pacejka backbone
    rear_c_kappa: float = 12.58710
    rear_lat_B: float = 11.25962
    rear_lat_C: float = 0.50000
    rear_lat_D: float = 8.89369
    rear_lat_E: float = 0.37898
    # rear tire, high slip: friction circle of radius rear_high_force, blended in at sigma_split
    rear_high_force: float = 4.60
    sigma_split: float = 0.300
    sigma_blend: float = 0.040


@dataclass(frozen=True)
class Limits:
    delta_max: float = math.radians(50.0)    # steering travel, symmetric
    omega_min: float = 0.0
    omega_max: float = 140.0
    ddelta_max: float = math.radians(500.0)  # input limits (rates of the references)
    domega_max: float = 600.0


def _magic(x, B, C, D, E):
    bx = B * x
    return D * jnp.sin(C * jnp.arctan(bx - E * (bx - jnp.arctan(bx))))


def _wrap(t):
    return jnp.arctan2(jnp.sin(t), jnp.cos(t))


def chassis(m: Vehicle, x5, rates):
    """d/dt [beta, r, V, delta, omega] given the steering and wheel-speed rates."""
    beta, r, V, delta, omega = x5[0], x5[1], x5[2], x5[3], x5[4]
    u, v = V * jnp.cos(beta), V * jnp.sin(beta)
    alpha_f = _wrap(jnp.arctan2(v + m.a_cg * r, u) - delta)
    alpha_r = _wrap(jnp.arctan2(v - m.b_cg * r, u))
    kappa = (m.wheel_radius * omega - u) / jnp.maximum(jnp.abs(u), 1e-8)
    fyf = -_magic(alpha_f, m.front_B, m.front_C, m.front_D, m.front_E)
    lat = jnp.tan(alpha_r)
    sig = jnp.sqrt(kappa ** 2 + lat ** 2 + 1e-18)
    s = jnp.maximum(sig, 1e-9)
    blend = 0.5 * (1.0 + jnp.tanh((sig - m.sigma_split) / m.sigma_blend))
    fxr = (1 - blend) * m.rear_c_kappa * kappa + blend * m.rear_high_force * kappa / s
    fyr = ((1 - blend) * (-_magic(alpha_r, m.rear_lat_B, m.rear_lat_C, m.rear_lat_D, m.rear_lat_E))
           + blend * (-m.rear_high_force * lat / s))
    fx = fxr - fyf * jnp.sin(delta)
    fy = fyr + fyf * jnp.cos(delta)
    vdot = (fx * jnp.cos(beta) + fy * jnp.sin(beta)) / m.mass
    bdot = (fy * jnp.cos(beta) - fx * jnp.sin(beta)) / (m.mass * jnp.maximum(V, 1e-6)) - r
    rdot = (m.a_cg * fyf * jnp.cos(delta) - m.b_cg * fyr) / m.izz
    return jnp.stack([bdot, rdot, vdot, rates[0], rates[1]])


def make_plant(m: Vehicle, kappa: float, servo_tau: float, drive_tau: float):
    """9-state dynamics f(x, u): the references move with u, the actuators chase them."""
    def f(x, u):
        rates = jnp.stack([(x[7] - x[5]) / servo_tau, (x[8] - x[6]) / drive_tau])
        x5d = chassis(m, x[2:7], rates)
        e, dphi, V = x[0], x[1], x[4]
        s_dot = V * jnp.cos(dphi) / (1.0 - kappa * e)
        e_dot = V * jnp.sin(dphi)
        dphi_dot = (x[3] + x5d[0]) - kappa * s_dot           # course rate minus path rate
        return jnp.concatenate([jnp.stack([e_dot, dphi_dot]), x5d, u])
    return f


@dataclass(frozen=True)
class Equilibrium:
    V: float
    radius: float
    beta: float
    r: float
    delta: float
    omega: float
    curvature: float

    def x9(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.beta, self.r, self.V, self.delta, self.omega,
                         self.delta, self.omega])


def solve_equilibrium(m: Vehicle, V: float, radius: float, turn: float = 1.0,
                      lim: Limits = Limits()) -> Equilibrium:
    """Counter-steered donut at speed V on a circle of the given radius: solve
    beta_dot = r_dot = V_dot = 0 for (beta, delta, omega) with r = turn * V / radius."""
    r_eq = turn * V / radius
    f = jax.jit(lambda z: chassis(m, jnp.array([z[0], r_eq, V, z[1], z[2]]), jnp.zeros(2))[:3])
    res = lambda z: np.asarray(f(jnp.asarray(z)))
    lo = np.array([math.radians(-70.0), -lim.delta_max, 1e-3])
    hi = np.array([math.radians(70.0), lim.delta_max, lim.omega_max])
    starts = [(-44.5, -27.8, 88.1), (-35, -15, 80), (-55, -40, 110), (-25, -5, 70)]
    for b, d, w in starts:
        z0 = np.clip([turn * math.radians(b), turn * math.radians(d), w], lo + 1e-9, hi - 1e-9)
        z = least_squares(res, z0, bounds=(lo, hi), method="trf", x_scale=np.array([0.5, 0.5, 50.0]),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000).x
        drifting = z[0] * turn < -math.radians(20.0) and z[1] * turn < 0.0   # countersteer, large slip
        if drifting and np.max(np.abs(res(z))) <= 1e-8:
            return Equilibrium(V, radius, float(z[0]), r_eq, float(z[1]), float(z[2]), turn / radius)
    raise RuntimeError(f"no counter-steered drift found at V = {V} m/s, R = {radius} m")
