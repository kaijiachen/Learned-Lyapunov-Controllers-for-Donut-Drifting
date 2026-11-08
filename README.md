# Learned Lyapunov Controllers for Donut Drifting

Code to train and evaluate the controllers in the paper on a model of a
1/10-scale car holding a donut drift (1.356 m/s on a 0.5 m circle).

| Case | Lyapunov function | Policy | Training |
|---|---|---|---|
| `quad` | quadratic, V = zᵀPz | NN | Lyapunov loss |
| `nonquad` | neural, V = ‖φ(z) − φ(0)‖² + ε‖z‖² | NN | Lyapunov loss |
| `lqr_nn` | quadratic, P starts at the Riccati solution | LQR + NN residual | Lyapunov loss |
| `dpc` | none | LQR + NN residual | LQR cost over 3 s rollouts (DPC baseline) |

The untrained 9-state LQR is always evaluated alongside as a reference.

## Files

- `model.py`: vehicle model (tires, chassis, path kinematics, actuator lags) and the drift equilibrium solver.
- `train.py`: training, evaluation and export for the four cases. All hyperparameters are constants at the top.
- Trained policies, experimental data and videos of the hardware trials: https://bit.ly/4iDxHoq

## Install

```
pip install -r requirements.txt
```

Tested with Python 3.13, JAX 0.11, NumPy 2.5 and SciPy 1.18, on CPU.

## Run

```
python train.py                        # all four cases, seed 0
python train.py --cases quad lqr_nn    # a subset
python train.py --cases quad --steps 300 --quick    # smoke test, about a minute
```

```
do python train.py --seed $s --out results/seed$s; done
```

Each run writes to `--out` (default `results/`):

- `comparison.md`: the comparison table (set volume, Vdot violations, local decay rate,
  policy gain, and closed-loop rollouts under delay and tire-force error)
- `comparison.json`: the same numbers
- `<case>/policy.npz`: the trained policy and V, for deployment

Runtime on a laptop CPU: a few minutes per Lyapunov case, and about 4 hours for `dpc`
(backpropagation through 300-step rollouts).

## Method in brief

The 9-state plant is x = [e, Δφ, β, r, V, δ, ω, δ_ref, ω_ref], with inputs the rates of
the two references. Learning uses z = (x − x\*)/s, with s the box half-widths.

**Lyapunov cases** minimize, over states sampled from the box,

    −0.03 log Vol{V ≤ 1} + 3 E[max(V̇/V + α + m, 0)] + 100 E_faces[max(1.1 − V, 0)]
      + max(α + m − ρ, 0) + 0.01 E‖∂(u/u_max)/∂z‖²

where α is half the LQR's guaranteed decay rate, m = 0.05, and ρ is the decay rate of
the linearization at x\*.

**DPC** minimizes the LQR cost Σ(zᵀQ̃z + uᵀRu)Δt over 3 s rollouts of the same model,
from z₀ uniform in half the box, with the same gain penalty and no terminal cost.

The loop delay (2 ticks of 10 ms) is folded into the actuator lags for training. The
evaluation rollouts apply the real lags and the transport delay separately.

## Notes

- Training is sensitive to floating-point rounding, so exact numbers can differ across
  JAX versions and CPUs, as they would for a different seed. Compare statistics across
  seeds, not individual digits.
- The V rows of the DPC column use the LQR's Riccati V, which is not a Lyapunov
  function of the DPC policy. Read them as diagnostics only.
- The donut can be changed with `--target-speed` and `--target-radius`.
