import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
from barrier_surgery import BarrierSatSurgery

# --- 1. The BRST Layer (Optimization Stiffness) ---
class BRSTLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Ghost fields acting on the WEIGHTS, not the data
        self.ghost_left = nn.Parameter(torch.zeros(out_features, out_features))
        self.ghost_right = nn.Parameter(torch.zeros(in_features, in_features))

    def forward(self, x):
        return self.linear(x)

    def brst_defect(self):
        """
        Enforces Axiom LS on the PARAMETER MANIFOLD.
        Forces the weight matrix to be well-conditioned (orthonormal-ish).
        This fixes the "Vanishing Gradient" and "Flat Valley" problems.
        """
        W = self.linear.weight

        # If W is tall (more outputs), W^T W should be I
        if W.shape[0] >= W.shape[1]:
            gram = torch.matmul(W.t(), W)
            target = torch.eye(W.shape[1], device=W.device)
        # If W is wide (more inputs), W W^T should be I
        else:
            gram = torch.matmul(W, W.t())
            target = torch.eye(W.shape[0], device=W.device)

        return torch.norm(gram - target) ** 2


# --- 2. The Hypo-VICReg Network ---
class HypoVICReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # HYPOSTRUCTURE: Dynamic barrier surgery for variance clipping
        self.barrier_surgery = BarrierSatSurgery(
            num_layers=1,
            base_epsilon=1.0,
            learnable=True,
            surgery_mode='linear',
            temporal_schedule='warmup',
            min_epsilon=0.5,
            max_epsilon=2.0,
        )

        # INTERFACE LAYERS (Standard Linear)
        # Allows diffeomorphism (bending/stretching) at the boundary
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # INTERNAL LAYERS (BRST Linear)
        # High-capacity "Bulk" that is structurally rigid (easy to optimize)
        self.hidden1 = BRSTLinear(hidden_dim, hidden_dim)
        self.hidden2 = BRSTLinear(hidden_dim, hidden_dim)

        # LATENT HEAD (Standard)
        self.head = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = self.relu(self.input_proj(x))

        # BRST Layers process the internal representation
        h = self.relu(self.hidden1(h))
        h = self.relu(self.hidden2(h))

        z = self.head(h)
        return z

    def compute_brst_loss(self):
        # We only penalize the internal physics
        return self.hidden1.brst_defect() + self.hidden2.brst_defect()


# --- 3. The VICReg Loss (Axiom Cap + Axiom LS) ---
def vicreg_loss(z, x, model, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0):
    """
    The 'Manifold Loss'. Shapes Z.
    """
    # 1. Invariance (Axiom LS - Smoothness)
    # Generate a "view" by adding noise (simulating local diffusion)
    z_prime = model(x + torch.randn_like(x) * 0.1)
    loss_inv = nn.functional.mse_loss(z, z_prime)

    # 2. Variance (Axiom Cap - Collapse Prevention)
    # Force std of each dimension to be 1 (prevents collapse to a point)
    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    loss_var = torch.mean(nn.functional.relu(1 - std_z))

    # 3. Covariance (Axiom Cap - Orthogonality)
    # Force dimensions to be decorrelated (prevents collapse to a line)
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1]-1, z.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    return lambda_inv * loss_inv + lambda_var * loss_var + lambda_cov * loss_cov


# --- 4. Training ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_samples = 2000
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.05)
    X = torch.FloatTensor(X).to(device)
    X = (X - X.mean(0)) / X.std(0)

    # Note: No Decoder! VICReg is non-generative (pure manifold learning)
    model = HypoVICReg(3, 128, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 7000

    # BRST Weight (Optimization Constraint)
    # We can keep this constant because it doesn't fight the geometry anymore
    # It just keeps the weight matrices healthy.
    lambda_brst = 0.01

    # Curriculum phases for BRST annealing
    # Gas phase: pure manifold learning, no stiffness
    # Liquid phase: gradual ramp-up of stiffness
    # Solid phase: full stiffness constraint
    gas_phase_end = int(epochs * 0.15)       # ~1050 epochs
    liquid_phase_end = int(epochs * 0.45)    # ~3150 epochs

    print("Training Hypo-VICReg with Curriculum...")
    print(f"Gas Phase: 0-{gas_phase_end} | Liquid Phase: {gas_phase_end}-{liquid_phase_end} | Solid Phase: {liquid_phase_end}-{epochs}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        z = model(X)

        # Manifold Loss (VICReg)
        loss_manifold = vicreg_loss(z, X, model)

        # Optimization Loss (BRST)
        loss_brst = model.compute_brst_loss()

        # LINEAR WARMUP for Stiffness (Axiom LS)
        # Don't enforce stiffness immediately. Let the manifold unfold first.
        if epoch < gas_phase_end:
            # Gas Phase: No stiffness, pure topology discovery
            current_lambda_brst = 0.0
        elif epoch < liquid_phase_end:
            # Liquid Phase: Gradual ramp from 0.0 to lambda_brst
            progress = (epoch - gas_phase_end) / (liquid_phase_end - gas_phase_end)
            current_lambda_brst = lambda_brst * progress
        else:
            # Solid Phase: Full stiffness constraint
            current_lambda_brst = lambda_brst

        total_loss = loss_manifold + (current_lambda_brst * loss_brst)

        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            phase = "Gas" if epoch < gas_phase_end else ("Liquid" if epoch < liquid_phase_end else "Solid")
            print(f"Epoch {epoch} [{phase}]: Manifold={loss_manifold.item():.4f} | BRST={loss_brst.item():.4f} | λ_brst={current_lambda_brst:.4f}")

    return model, X, t


# --- 5. Visualize ---
def visualize(model, X, t):
    model.eval()
    with torch.no_grad():
        z = model(X).cpu().numpy()
        X_cpu = X.cpu().numpy()

    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_cpu[:, 0], X_cpu[:, 1], X_cpu[:, 2], c=t, cmap='Spectral', s=5)
    ax1.set_title("Swiss Roll")

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(z[:, 0], z[:, 1], c=t, cmap='Spectral', s=5)
    ax2.set_title("Hypo-VICReg Embedding\n(BRST Optimized)")

    plt.tight_layout()
    plt.savefig('hypo_vicreg.png', dpi=150)
    print("Saved visualization to hypo_vicreg.png")


if __name__ == "__main__":
    m, d, c = train()
    visualize(m, d, c)
