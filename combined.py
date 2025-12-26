import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from barrier_surgery import BarrierSatSurgery

# --- 1. The BRST Layer (Internal Stiffness) ---
class BRSTLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

    def brst_defect(self):
        # Forces weights to be near-orthogonal (Stiffness)
        W = self.linear.weight
        if W.shape[0] >= W.shape[1]:
            gram = torch.matmul(W.t(), W)
            target = torch.eye(W.shape[1], device=W.device)
        else:
            gram = torch.matmul(W, W.t())
            target = torch.eye(W.shape[0], device=W.device)
        return torch.norm(gram - target) ** 2

# --- 2. The Universal Hypostructure Network ---
class HypoUniversal(nn.Module):
    def __init__(self, input_dim, latent_dim, num_charts=3):
        super().__init__()
        self.num_charts = num_charts

        # HYPOSTRUCTURE: Dynamic barrier surgery for chart separation
        self.barrier_surgery = BarrierSatSurgery(
            num_layers=1,  # Single layer for separation
            base_epsilon=4.0,
            learnable=True,
            surgery_mode='sigmoid',
            temporal_schedule='warmup',
            min_epsilon=2.0,
            max_epsilon=10.0,
        )

        # A. The Router (Topology / Axiom TB)
        # Standard layers are fine here; cuts don't need to be isometric
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_charts),
            nn.Softmax(dim=1)
        )

        # B. The Experts (Geometry / Axiom LS)
        # Each chart is a BRST Network to ensure it unrolls cleanly
        self.charts = nn.ModuleList()
        for _ in range(num_charts):
            expert = nn.Sequential(
                BRSTLinear(input_dim, 128),
                nn.ReLU(),
                BRSTLinear(128, 128),
                nn.ReLU(),
                BRSTLinear(128, latent_dim)
            )
            self.charts.append(expert)

    def forward(self, x):
        weights = self.router(x) # [batch, num_charts]

        z = torch.zeros(x.size(0), 2, device=x.device)

        # We need individual chart outputs for the Separation Loss
        chart_outputs = []

        for i in range(self.num_charts):
            z_i = self.charts[i](x)
            chart_outputs.append(z_i)
            z += weights[:, i:i+1] * z_i

        return z, weights, chart_outputs

    def compute_brst_loss(self):
        # Sum defect over all layers in all charts
        total_defect = 0
        for chart in self.charts:
            for layer in chart:
                if isinstance(layer, BRSTLinear):
                    total_defect += layer.brst_defect()
        return total_defect

# --- 3. The Grand Unified Loss ---
def universal_loss(z, x, weights, chart_outputs, model):
    # 1. VICReg (Data Manifold)
    z_prime, _, _ = model(x + torch.randn_like(x) * 0.05)
    loss_inv = nn.functional.mse_loss(z, z_prime)
    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    loss_var = torch.mean(nn.functional.relu(1 - std_z))
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1]-1, z.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    # 2. Topology (Router Constraints)
    loss_entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-6), dim=1))

    mean_usage = torch.mean(weights, dim=0)
    target_usage = torch.tensor([1.0 / model.num_charts] * model.num_charts, device=x.device)
    loss_balance = torch.norm(mean_usage - target_usage) ** 2

    # 3. Separation (Force charts apart)
    loss_sep = torch.tensor(0.0, device=x.device)
    centers = []
    for i in range(model.num_charts):
        # Weighted mean of this chart's output
        w_i = weights[:, i:i+1]
        if w_i.sum() > 0:
            center = (chart_outputs[i] * w_i).sum(dim=0) / (w_i.sum() + 1e-6)
            centers.append(center)

    if len(centers) > 1:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = torch.norm(centers[i] - centers[j]).unsqueeze(0)  # Make it 1D

                # HYPOSTRUCTURE: Dynamic barrier clipping (replaces fixed epsilon=4.0)
                # Using barrier saturation surgery for adaptive separation threshold
                loss_sep += model.barrier_surgery(dist, layer_idx=0).squeeze()

    # 4. BRST (Internal Stiffness)
    loss_brst = model.compute_brst_loss()

    return (25.0 * loss_inv) + (25.0 * loss_var) + (1.0 * loss_cov) + \
        (2.0 * loss_entropy) + (100.0 * loss_balance) + (10.0 * loss_sep) + \
        (0.01 * loss_brst)

# --- 4. The Nightmare Dataset ---
def get_nightmare_data(n_samples=3000):
    n_per = n_samples // 3

    # 1. Swiss Roll
    X1, _ = make_swiss_roll(n_per, noise=0.05)
    X1 = (X1 - X1.mean(0)) / (X1.std(0)  + 1e-6 )
    X1[:, 0] -= 4.0 # Shift Left
    t1 = np.zeros(n_per) # Label 0

    # 2. Sphere
    phi = np.random.uniform(0, 2*np.pi, n_per)
    costheta = np.random.uniform(-1, 1, n_per)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    X2 = np.stack([x, y, z], axis=1)
    X2[:, 0] += 0.0 # Center
    t2 = np.ones(n_per) # Label 1

    # 3. Moons (embedded in 3D)
    X3_2d, _ = make_moons(n_per, noise=0.05)
    X3 = np.zeros((n_per, 3))
    X3[:, 0] = X3_2d[:, 0]
    X3[:, 1] = X3_2d[:, 1]
    X3 = (X3 - X3.mean(0)) /( X3.std(0)  + 1e-6 )
    X3[:, 0] += 4.0 # Shift Right
    t3 = np.full(n_per, 2) # Label 2

    X = np.vstack([X1, X2, X3])
    t = np.concatenate([t1, t2, t3])

    return torch.FloatTensor(X), t

# --- 5. Training ---
def train_universal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, labels = get_nightmare_data(3000)
    X = X.to(device)

    # 3 Charts: Hopefully one for Roll, one for Sphere, one for Moons?
    # Or maybe it splits the sphere into two and combines Moons/Roll?
    # Let's give it 4 charts to be safe (Sphere needs 2, Roll needs 1, Moons needs 1)
    model = HypoUniversal(input_dim=3, latent_dim=2, num_charts=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 8000

    print("Training Universal Hypostructure...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, weights, charts = model(X)

        loss = universal_loss(z, X, weights, charts, model)
        loss.backward()
        optimizer.step()

        # HYPOSTRUCTURE: Update barrier surgery temporal schedule
        model.barrier_surgery.step_schedule()

        if epoch % 1000 == 0:
            usage = weights.mean(dim=0).detach().cpu().numpy()

            # HYPOSTRUCTURE: Get current barrier epsilon
            barrier_stats = model.barrier_surgery.get_epsilon_stats()
            epsilon_current = barrier_stats['epsilons'][0]

            print(f"Epoch {epoch}: Loss={loss.item():.4f} | ε_sep={epsilon_current:.2f} | Usage={usage}")

    return model, X, labels

# --- 6. Visualization ---
def visualize_universal(model, X, labels):
    model.eval()
    with torch.no_grad():
        z, weights, _ = model(X)
        z = z.cpu().numpy()
        X_cpu = X.cpu().numpy()
        weights = weights.cpu().numpy()

    chart_choice = np.argmax(weights, axis=1)

    plt.figure(figsize=(18, 6))

    # Input
    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.scatter(X_cpu[:,0], X_cpu[:,1], X_cpu[:,2], c=labels, cmap='Set1', s=2)
    ax1.set_title("Input: The Nightmare\n(Roll, Sphere, Moons)")

    # Latent (colored by ground truth identity)
    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(z[:,0], z[:,1], c=labels, cmap='Set1', s=2)
    ax2.set_title("Latent Space\n(Separation by Identity)")

    # Latent (colored by Chart usage)
    ax3 = plt.subplot(1, 3, 3)
    scatter = ax3.scatter(z[:,0], z[:,1], c=chart_choice, cmap='tab10', s=2)
    ax3.set_title("Structural Surgery\n(Colors = Different Charts)")
    plt.colorbar(scatter, ax=ax3, ticks=range(model.num_charts))

    plt.tight_layout()
    plt.savefig('hypo_universal.png', dpi=150)
    print("Saved hypo_universal.png")

if __name__ == "__main__":
    m, d, c = train_universal()
    visualize_universal(m, d, c)
