import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from barrier_surgery import BarrierSatSurgery

# --- 1. The Hypo-Atlas Architecture (Deeper Experts) ---
class HypoAtlas(nn.Module):
    def __init__(self, input_dim, latent_dim, num_charts=2):
        super().__init__()
        self.num_charts = num_charts

        # HYPOSTRUCTURE: Dynamic barrier surgery for chart separation
        self.barrier_surgery = BarrierSatSurgery(
            num_layers=1,  # Single layer for separation
            base_epsilon=3.0,
            learnable=True,
            surgery_mode='sigmoid',
            temporal_schedule='warmup',
            min_epsilon=1.0,
            max_epsilon=8.0,
        )

        # The Router (Topology Learner)
        # Determines the "Atlas" structure
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_charts),
            nn.Softmax(dim=1)
        )

        # The Charts (Geometry Learners)
        # We make them deeper so they can learn non-linear "unwrapping" (like stereographic projection)
        self.charts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim)
            ) for _ in range(num_charts)
        ])

    def forward(self, x):
        # weights: [batch, num_charts]
        weights = self.router(x)

        z = torch.zeros(x.size(0), 2, device=x.device)

        # Apply the atlas mapping: z = sum( w_i * f_i(x) )
        for i in range(self.num_charts):
            z_i = self.charts[i](x)
            z += weights[:, i:i+1] * z_i

        return z, weights


# --- 2. The Structural Loss (With Capacity Constraint + Sector Separation) ---
def atlas_loss(z, x, weights, model):
    # --- A. Reconstruction (Standard VICReg) ---
    # 1. Invariance (Local Stiffness): Local points stay local
    z_prime, _ = model(x + torch.randn_like(x) * 0.05)
    loss_inv = nn.functional.mse_loss(z, z_prime)

    # 2. Variance (Prevent point collapse)
    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    loss_var = torch.mean(nn.functional.relu(1 - std_z))

    # 3. Covariance (Prevent dimensional collapse)
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1]-1, z.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    # --- B. Topological Constraints ---

    # 1. Entropy (Axiom TB): Force hard decisions.
    # We want weights to be close to 0 or 1, not 0.5.
    loss_entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-6), dim=1))

    # 2. Balance (Axiom Cap): Force equal chart usage.
    # Prevents Mode Collapse (one chart taking over).
    mean_usage = torch.mean(weights, dim=0)
    target_usage = torch.tensor([1.0 / model.num_charts] * model.num_charts, device=x.device)
    loss_balance = torch.norm(mean_usage - target_usage) ** 2

    # --- C. NEW: Sector Separation (The Barrier) ---
    # Calculate the "center of mass" for each chart's contribution
    # and force them away from each other.
    # This prevents Topological Superposition (Mode T.E).

    # Approximate center of Chart 0 vs Chart 1 based on high-confidence points
    mask0 = weights[:, 0] > 0.5
    mask1 = weights[:, 1] > 0.5

    if mask0.sum() > 0 and mask1.sum() > 0:
        center0 = z[mask0].mean(dim=0)
        center1 = z[mask1].mean(dim=0)
        # We want to MAXIMIZE distance, so we MINIMIZE negative distance
        dist = torch.norm(center0 - center1).unsqueeze(0)  # Make it 1D for barrier surgery

        # HYPOSTRUCTURE: Dynamic barrier clipping (replaces fixed epsilon=3.0)
        # Using barrier saturation surgery for adaptive separation threshold
        loss_separation = model.barrier_surgery(dist, layer_idx=0).squeeze()
    else:
        loss_separation = torch.tensor(0.0, device=x.device)

    # Weights
    return (25.0 * loss_inv) + (25.0 * loss_var) + (1.0 * loss_cov) + \
           (2.0 * loss_entropy) + (100.0 * loss_balance) + (10.0 * loss_separation)


# --- 3. Data: Sphere ---
def get_sphere_data(n_samples=2000):
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    costheta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(costheta)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    data = np.stack([x, y, z], axis=1)
    colors = z
    return torch.FloatTensor(data), colors


# --- 4. Training ---
def train_atlas():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    X, colors = get_sphere_data(4000)  # More points helps density estimation
    X = X.to(device)

    model = HypoAtlas(input_dim=3, latent_dim=2, num_charts=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)  # Slightly higher LR

    epochs = 6000  # More epochs to let the charts settle

    print("Training Hypo-Atlas on Sphere...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, weights = model(X)

        loss = atlas_loss(z, X, weights, model)
        loss.backward()
        optimizer.step()

        # HYPOSTRUCTURE: Update barrier surgery temporal schedule
        model.barrier_surgery.step_schedule()

        if epoch % 1000 == 0:
            # Monitor balance specifically
            usage = weights.mean(dim=0).detach().cpu().numpy()

            # HYPOSTRUCTURE: Get current barrier epsilon
            barrier_stats = model.barrier_surgery.get_epsilon_stats()
            epsilon_current = barrier_stats['epsilons'][0]

            print(f"Epoch {epoch}: Loss={loss.item():.4f} | ε_sep={epsilon_current:.2f} | Chart Usage: {usage}")

    return model, X, colors


# --- 5. Visualization ---
def visualize_atlas(model, X, colors):
    model.eval()
    with torch.no_grad():
        z, weights = model(X)
        z = z.cpu().numpy()
        X_cpu = X.cpu().numpy()
        weights = weights.cpu().numpy()

    chart_choice = np.argmax(weights, axis=1)

    plt.figure(figsize=(16, 5))

    # 1. Learned Cut on Sphere
    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.scatter(X_cpu[:,0], X_cpu[:,1], X_cpu[:,2], c=chart_choice, cmap='bwr', s=5)
    ax1.set_title("Axiom TB: The Learned Cut\n(Should look like hemispheres)")

    # 2. Latent Space (Global View)
    ax2 = plt.subplot(1, 3, 2)
    # We want to see if they separated.
    ax2.scatter(z[:,0], z[:,1], c=colors, cmap='coolwarm', s=2)
    ax2.set_title("Global Latent Space\n(Red=North, Blue=South)")

    # 3. Latent Space (Chart View)
    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(z[:,0], z[:,1], c=chart_choice, cmap='bwr', s=2)
    ax3.set_title("Chart Separation\n(Red=Chart 0, Blue=Chart 1)")

    plt.tight_layout()
    plt.savefig('hypo_atlas_v2.png', dpi=150)
    print("Saved hypo_atlas_v2.png")


if __name__ == "__main__":
    m, d, c = train_atlas()
    visualize_atlas(m, d, c)
