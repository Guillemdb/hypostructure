import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from tqdm import tqdm

# ... [Include BRSTLinear and HypoNatural classes from previous code] ...
# (They remain exactly the same)

class BRSTLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x): return self.linear(x)
    def brst_defect(self):
        W = self.linear.weight
        if W.shape[0] >= W.shape[1]:
            gram = torch.matmul(W.t(), W)
            target = torch.eye(W.shape[1], device=W.device)
        else:
            gram = torch.matmul(W, W.t())
            target = torch.eye(W.shape[0], device=W.device)
        return torch.norm(gram - target) ** 2

class HypoDiscovery(nn.Module):
    def __init__(self, input_dim, latent_dim, max_charts=10, decay=0.99):  # Slower decay
        super().__init__()
        self.max_charts = max_charts
        self.decay = decay
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Buffer for history
        self.register_buffer('avg_usage', torch.ones(max_charts) / max_charts)

        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_charts)
        )
        self._init_router()

        self.charts = nn.ModuleList()
        for _ in range(max_charts):
            self.charts.append(self._make_expert())

    def _make_expert(self):
        return nn.Sequential(
            BRSTLinear(self.input_dim, 128),
            nn.ReLU(),
            BRSTLinear(128, 128),
            nn.ReLU(),
            BRSTLinear(128, self.latent_dim)
        )

    def _init_router(self):
        # Initialize flat to encourage equality start
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.router[-1].bias, 0.0)

    def forward(self, x):
        logits = self.router(x)
        # Higher temperature (3.0) keeps gradients flowing even to smaller logits
        weights = nn.functional.gumbel_softmax(logits, tau=3.0, hard=False, dim=1)

        # Update global usage stats (detached - for monitoring only)
        if self.training:
            batch_usage = weights.mean(dim=0).detach()
            self.avg_usage = self.decay * self.avg_usage + (1 - self.decay) * batch_usage

        z = torch.zeros(x.size(0), 2, device=x.device)
        chart_outputs = []
        for i in range(self.max_charts):
            z_i = self.charts[i](x)
            chart_outputs.append(z_i)
            z += weights[:, i:i+1] * z_i
        return z, weights, chart_outputs

    def compute_brst_loss(self):
        total = 0
        for chart in self.charts:
            for layer in chart:
                if isinstance(layer, BRSTLinear):
                    total += layer.brst_defect()
        return total

    # --- THE LAZARUS MECHANISM (Axiom Rec) ---
    def revive_dead_charts(self, threshold=0.02):
        """
        Detects dead charts and resets their weights.
        This forces the system to 'retry' using these resources.
        """
        dead_indices = (self.avg_usage < threshold).nonzero(as_tuple=True)[0]

        if len(dead_indices) > 0:
            for idx in dead_indices:
                # 1. Reset Router weights for this chart (give it a new random direction)
                nn.init.normal_(self.router[-1].weight[idx], mean=0.0, std=0.1)
                nn.init.constant_(self.router[-1].bias[idx], 0.0)

                # 2. Reset the Expert itself (give it a fresh brain)
                self.charts[idx] = self._make_expert().to(self.avg_usage.device)

                # 3. Reset usage stats so it doesn't get killed immediately next step
                self.avg_usage[idx] = 1.0 / self.max_charts

# --- 3. Per-Chart VICReg Helper ---
def per_chart_vicreg(z_i, weights_i, noise_z_i):
    """Compute weighted VICReg loss for a single chart."""
    w_sum = weights_i.sum()
    if w_sum < 1.0:
        return torch.tensor(0.0, device=z_i.device)

    # Weighted invariance
    loss_inv = (weights_i * (z_i - noise_z_i).pow(2).sum(dim=1)).sum() / w_sum

    # Weighted variance (using weighted mean/std)
    w_norm = weights_i / w_sum
    weighted_mean = (w_norm.unsqueeze(1) * z_i).sum(dim=0)
    weighted_var = (w_norm.unsqueeze(1) * (z_i - weighted_mean).pow(2)).sum(dim=0)
    std_z = torch.sqrt(weighted_var + 1e-04)
    loss_var = torch.mean(torch.relu(1 - std_z))

    # Weighted covariance
    z_centered = z_i - weighted_mean
    weighted_cov = (w_norm.unsqueeze(1) * z_centered).T @ z_centered
    off_diag = weighted_cov.flatten()[:-1].view(z_i.shape[1]-1, z_i.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    return 25.0*loss_inv + 25.0*loss_var + 1.0*loss_cov

# --- 4. Load Balancing Loss (from Switch Transformers) ---
def load_balance_loss(weights, num_charts):
    """
    Encourages each chart to receive equal fraction of samples.
    Minimized when load is balanced AND probabilities match assignments.
    """
    # f_i: fraction of samples routed to each expert (hard assignment)
    # Vectorized one-hot counting
    hard_assignments = weights.argmax(dim=1)
    f = torch.zeros(num_charts, device=weights.device)
    f.scatter_add_(0, hard_assignments, torch.ones_like(hard_assignments, dtype=torch.float))
    f = f / weights.size(0)

    # p_i: average routing probability
    p = weights.mean(dim=0)

    # Auxiliary loss: penalizes imbalanced routing
    return num_charts * (f * p).sum()

# --- 5. The Natural Loss (With Extended Heating/Cooling Schedule) ---
def natural_loss(z, x, weights, chart_outputs, model, epoch):
    # --- PHASE SCHEDULE (Extended Heating) ---
    if epoch < 200:
        # HEATING PHASE: Strong reward for diversity
        entropy_coeff = -50.0
        local_entropy_coeff = 0.0  # No hard decisions during heating
    elif epoch < 400:
        # MAINTENANCE PHASE: Weak diversity pressure
        entropy_coeff = -5.0
        local_entropy_coeff = 2.0  # Gradual hardening
    else:
        # COOLING PHASE: Force Selection
        entropy_coeff = min(2.0, 0.02 * (epoch - 400))
        local_entropy_coeff = 5.0  # Full hard decisions

    # A. Global Geometry (on mixed output)
    noise_x = x + torch.randn_like(x) * 0.05
    z_prime, _, noise_chart_outputs = model(noise_x)
    loss_inv = nn.functional.mse_loss(z, z_prime)

    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    loss_var = torch.mean(nn.functional.relu(1 - std_z))

    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1]-1, z.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    loss_geo = 100.0*loss_inv + 100.0*loss_var + 5.0*loss_cov

    # A2. Per-Chart Geometry (forces each chart to specialize)
    loss_per_chart = torch.tensor(0.0, device=x.device)
    for i in range(model.max_charts):
        w_i = weights[:, i]
        if w_i.sum() > 1.0:  # Chart has enough responsibility
            loss_per_chart = loss_per_chart + per_chart_vicreg(
                chart_outputs[i], w_i, noise_chart_outputs[i]
            )

    # B. Parsimony (The Thermostat)
    history = model.avg_usage
    # Entropy of the usage distribution
    global_entropy = -torch.sum(history * torch.log(history + 1e-6))
    metabolic_cost = entropy_coeff * global_entropy

    # C. Local Certainty (Action Gap)
    local_entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-6), dim=1))

    # D. Separation (Vectorized)
    loss_sep = torch.tensor(0.0, device=x.device)

    # 1. Identify valid charts: history > 0.01 AND batch weight sum > 1e-3
    batch_counts = weights.sum(dim=0)
    valid_mask = (model.avg_usage > 0.01) & (batch_counts > 1e-3)

    if valid_mask.sum() > 1:
        # 2. Stack chart outputs: [Batch, Num_Charts, Latent_Dim]
        z_stack = torch.stack(chart_outputs, dim=1)

        # 3. Compute weighted centers for ALL charts
        # weights: [Batch, Num_Charts] -> [Batch, Num_Charts, 1]
        w_expanded = weights.unsqueeze(-1)
        weighted_sum = (z_stack * w_expanded).sum(dim=0)  # [Num_Charts, Latent_Dim]
        all_centers = weighted_sum / (batch_counts.unsqueeze(-1) + 1e-6)

        # 4. Filter to valid charts only: [N_Valid, Latent_Dim]
        valid_centers = all_centers[valid_mask]

        # 5. Pairwise distances: [N_Valid, N_Valid]
        dists = torch.cdist(valid_centers, valid_centers, p=2)

        # 6. Hinge loss on upper triangle (i < j pairs only)
        hinge_matrix = torch.relu(5.0 - dists)
        loss_sep = torch.triu(hinge_matrix, diagonal=1).sum()

    # E. Balance Loss (on BATCH weights - HAS GRADIENT to router!)
    batch_usage_for_loss = weights.mean(dim=0)  # NOT detached - in computation graph
    target_usage = 1.0 / model.max_charts
    loss_balance = (batch_usage_for_loss - target_usage).pow(2).sum()

    # F. Load Balancing Loss (from Switch Transformers)
    loss_load = load_balance_loss(weights, model.max_charts)

    # G. BRST Regularization
    loss_brst = model.compute_brst_loss()

    return (
        loss_geo +                              # Global geometry
        loss_per_chart +                        # Per-chart geometry
        metabolic_cost +                        # Entropy thermostat
        local_entropy_coeff * local_entropy +   # Hard decisions (scheduled)
        10.0 * loss_sep +                       # Separation
        100.0 * loss_balance +                  # Balance (has gradient now!)
        1.0 * loss_load +                       # Load balancing (increased)
        0.01 * loss_brst                        # BRST regularization
    )

def get_data(n_samples=3000):
    # (Same data generation as before)
    n = n_samples // 3
    X1, _ = make_swiss_roll(n, noise=0.05)
    X1 = (X1 - X1.mean(0)) / (X1.std(0) + 1e-6)
    X1[:, 0] -= 4.0
    t1 = np.zeros(n)
    phi = np.random.uniform(0, 2*np.pi, n)
    ct = np.random.uniform(-1, 1, n)
    th = np.arccos(ct)
    X2 = np.stack([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)], 1)
    X2 = (X2 - X2.mean(0)) / (X2.std(0)+1e-6)
    t2 = np.ones(n)
    X3_2d, _ = make_moons(n, noise=0.05)
    X3 = np.zeros((n, 3))
    X3[:, :2] = X3_2d
    mean = X3.mean(0)
    std = X3.std(0)
    std[std==0] = 1.0
    X3 = (X3 - mean) / std
    X3[:, 0] += 4.0
    t3 = np.full(n, 2)
    return torch.FloatTensor(np.vstack([X1, X2, X3])), np.concatenate([t1, t2, t3])

# --- 4. The Minibatch Training Loop ---
def run_minibatch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_full, labels_full = get_data(6000)
    dataset = TensorDataset(X_full)
    BATCH_SIZE = 64
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 10 Charts
    model = HypoDiscovery(3, 2, max_charts=10).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)

    epochs = 200  # Extended for new phase schedule

    print(f"Starting Discovery with Extended Heating + Per-Chart Geometry...")

    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0

        for batch_idx, (x_batch,) in enumerate(dataloader):
            x_batch = x_batch.to(device)

            opt.zero_grad()
            z, w, charts_out = model(x_batch)
            loss = natural_loss(z, x_batch, w, charts_out, model, epoch)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        # --- LAZARUS STEP ---
        # Revive through heating AND maintenance phases (0-300)
        if epoch < 300/800 * epochs:
            model.revive_dead_charts(threshold=0.02)

        if epoch % 20 == 0:
            usage = model.avg_usage.detach().cpu().numpy()
            active = np.sum(usage > 0.05)
            usage_str = " ".join([f"{u:.2f}" for u in usage])
            tqdm.write(f"Epoch {epoch}: Loss={epoch_loss/len(dataloader):.2f} | Active={active} | Usage=[{usage_str}]")

    # 3. Vis (Use full dataset)
    model.eval()
    X_vis = X_full.to(device)
    z = model(X_vis)[0].detach().cpu().numpy()
    w = model(X_vis)[1].detach().cpu().numpy()
    charts = np.argmax(w, axis=1)

    u_charts = np.unique(charts)
    remap = {old: new for new, old in enumerate(u_charts)}
    charts_mapped = np.array([remap[c] for c in charts])

    X_cpu = X_full.numpy()

    plt.figure(figsize=(18, 6))

    # 1. Input colored by dataset
    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.scatter(X_cpu[:,0], X_cpu[:,1], X_cpu[:,2], c=labels_full, cmap='Set1', s=3)
    ax1.set_title("Input (colored by dataset)")

    # 2. Embedding colored by dataset
    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(z[:,0], z[:,1], c=labels_full, cmap='Set1', s=3)
    ax2.set_title("Embedding (colored by dataset)")
    ax2.set_aspect('equal')

    # 3. Embedding colored by chart assignment
    ax3 = plt.subplot(1, 3, 3)
    sc = ax3.scatter(z[:,0], z[:,1], c=charts_mapped, cmap='tab20', s=3)
    ax3.set_title(f"Embedding (colored by chart): {len(u_charts)} Charts")
    ax3.set_aspect('equal')
    plt.colorbar(sc, ax=ax3)

    plt.tight_layout()
    plt.savefig("hypo_minibatch.png", dpi=150)
    print("Saved hypo_minibatch.png")

if __name__ == "__main__":
    run_minibatch()