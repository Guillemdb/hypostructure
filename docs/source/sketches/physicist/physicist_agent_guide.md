# Physicist Agent: Architecture & Training Guide

## Overview

The **Physicist Agent** is a reinforcement learning agent that learns physics-aware representations for continuous control. It combines:

1. **Split-Brain VAE**: Separates observations into physics-relevant (z_macro) and noise (z_micro) components
2. **World Model**: Predicts future states using only z_macro (causal enclosure)
3. **PPO Policy**: Standard proximal policy optimization
4. **Lyapunov Stability**: Geometric constraints on the value function

The key insight is that **physics should be predictable from a low-dimensional state**, while noise and irrelevant details can be discarded.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           PHYSICIST AGENT               │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENCODER (Split-Brain VAE)                       │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐   ┌─────────────┐  │
│  │  obs    │───▶│  Shared MLP  │───▶│  z_macro    │   │  z_micro    │  │
│  │ (dim=N) │    │  (256 hidden)│    │  (dim=16)   │   │  (dim=32)   │  │
│  └─────────┘    └──────────────┘    │  Physics    │   │  Noise      │  │
│                                     │  State      │   │  Encoding   │  │
│                                     └──────┬──────┘   └──────┬──────┘  │
└────────────────────────────────────────────┼─────────────────┼─────────┘
                                             │                 │
                      ┌──────────────────────┼─────────────────┤
                      │                      │                 │
                      ▼                      ▼                 ▼
          ┌───────────────────┐    ┌─────────────────────────────────┐
          │   WORLD MODEL     │    │     ACTOR (Policy Network)      │
          │   (PhysicsEngine) │    │  (z_macro, z_micro) → π(a|z)    │
          │                   │    └─────────────────────────────────┘
          │  z_macro_t ──────▶│──▶ z_macro_{t+1}                     │
          │                   │                                       │
          │  BLIND to z_micro │    ┌─────────────────────────────────┐
          └───────────────────┘    │     CRITIC (Value Network)      │
                                   │  (z_macro, z_micro) → V(z)      │
                                   │  Also used for Lyapunov ∇V      │
                                   └─────────────────────────────────┘
```

### Component Details

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Encoder** | obs (N-dim) | z_macro (16), z_micro (32) | Split physics from noise |
| **World Model** | z_macro_t | z_macro_{t+1} | Predict future physics state |
| **Actor** | z_macro, z_micro | action distribution | Policy for control |
| **Critic** | z_macro, z_micro | V(s) scalar | Value estimation + Lyapunov |

---

## Loss Functions (14 Total)

### PPO Losses (3)

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| Policy | clip(ratio × A, 1±ε) | - | Stable policy updates |
| Value | (V - G)² | 0.5 | Value function fitting |
| Entropy | -H(π) | 0.01 | Exploration |

### Physicist Losses (5)

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Closure** | ‖WM(z_m^t) - z_m^{t+1}‖² | 1.0 | World model accuracy |
| **Slowness** | ‖z_m^t - z_m^{t-1}‖² | 0.1 | Temporal smoothness |
| **KL Micro** | KL(q(z_μ) ‖ N(0,I)) | 0.01 | Push noise to prior |
| **KL Macro** | KL(q(z_m) ‖ N(0,I)) | 0.001 | Light regularization |
| **BRST** | ‖W^T W - I‖² | 0.001 | Orthogonal weights |

### Lyapunov & Geometric Losses (3)

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Lyapunov** | max(0, V̇ + αV)² | 1.0 | Exponential stability |
| **Eikonal** | (‖∇V‖ - 1)² | 0.1 | Valid distance function |
| **Stiffness** | max(0, ε - ‖∇V‖)² | 0.01 | Non-vanishing gradients |

### Synchronization Losses (3)

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Sync** | ‖z_enc^{t+1} - sg(WM)‖² | 0.1 | VAE-WM coherence |
| **Zeno** | KL(π_t ‖ π_{t-1}) | 0.1 | Action smoothness |
| **VICReg** | inv + var + cov | 1.0 | Collapse prevention |

---

## Key Theoretical Principles

### 1. Causal Enclosure

The world model only sees z_macro, not z_micro. This forces:
- z_macro to contain all information needed for prediction
- z_micro to capture only noise/irrelevant details

```python
# World model is BLIND to z_micro
z_next = world_model(z_macro)  # No z_micro input!
```

### 2. Lyapunov Stability

The value function V(s) acts as a Lyapunov function:
- V̇(s) ≤ -αV(s) ensures exponential convergence
- States flow "downhill" toward goals

```
V̇ = ∇V · ṡ  where ṡ = WM(z) - z
Constraint: V̇ + αV ≤ 0
```

### 3. Eikonal Regularization

Force ‖∇V‖ ≈ 1 so V represents a valid geodesic distance:
- Like a signed distance function in graphics
- Ensures consistent "distance to goal" semantics

### 4. Ruppeiner Metric

Extract loss landscape geometry from Adam's second moment:
```python
g_ij ≈ E[∂L/∂θ_i · ∂L/∂θ_j] ≈ v_i  # Adam's exp_avg_sq
```
- **Condition number κ**: Ratio of max/min curvature (lower = better)
- **Flatness**: Inverse of average curvature (higher = flatter minima)

---

## Training Dynamics

### Typical Loss Evolution

| Phase | Updates | What Happens |
|-------|---------|--------------|
| **Early** | 0-50 | High closure/slowness, BRST stabilizing |
| **Mid** | 50-200 | Closure drops, reward increases |
| **Late** | 200+ | Losses plateau, reward variance |

### Healthy Training Signs

- ✓ Closure decreasing (world model learning)
- ✓ Slowness decreasing (smooth latents)
- ✓ Eikonal < 0.1 (valid distance function)
- ✓ Zeno < 0.05 (smooth policy)
- ✓ VICReg stable 10-20 (no collapse)
- ✓ Reward trending up

### Warning Signs

- ⚠ BRST increasing: Orthogonality constraint failing
- ⚠ Closure increasing: World model diverging
- ⚠ VICReg → 0: Representation collapse
- ⚠ Lyapunov spiking: Stability violations

---

## Usage

### Basic Training

```bash
python src/experiments/physicist/physicist_agent_continuous.py \
  --env_id Ant-v5 \
  --num_envs 16 \
  --total_timesteps 1000000
```

### Recommended Settings by Environment

| Environment | num_envs | timesteps | Notes |
|-------------|----------|-----------|-------|
| InvertedPendulum-v5 | 8 | 200k | Easy, fast convergence |
| InvertedDoublePendulum-v5 | 16 | 500k | Moderate difficulty |
| HalfCheetah-v5 | 16 | 2M | Needs longer training |
| Ant-v5 | 32 | 5M | Complex, high-dim |
| Humanoid-v5 | 64 | 10M | Most challenging |

### Tuning Loss Weights

```bash
# Increase Lyapunov for more stability
--lambda_lyapunov 2.0

# Reduce BRST if it's diverging
--lambda_brst 0.0001

# Increase VICReg if representations collapse
--lambda_vicreg 2.0
```

---

## Interpreting Metrics

### Ruppeiner Condition Number (κ)

| Value | Interpretation |
|-------|----------------|
| < 10,000 | Excellent conditioning |
| 10k - 100k | Good |
| 100k - 1M | Typical for deep networks |
| > 1M | May have optimization issues |

### Ruppeiner Flatness

| Value | Interpretation |
|-------|----------------|
| > 1e7 | Very flat (good generalization) |
| 1e5 - 1e7 | Normal |
| < 1e5 | Sharp minima (may overfit) |

---

## File Structure

```
src/experiments/physicist/
├── physicist_agent_continuous.py  # Main agent + training loop
├── losses.py                      # Loss functions (VICReg, Lyapunov, etc.)
├── layers.py                      # BRST linear layers
└── __init__.py
```

---

## References

- **Fragile Framework**: `docs/source/sketches/fragile/fragile-index.md`
- **VICReg**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization"
- **Lyapunov RL**: Berkenkamp et al., "Safe Model-based Reinforcement Learning"
- **Ruppeiner Geometry**: Ruppeiner, "Riemannian geometry in thermodynamic fluctuation theory"
