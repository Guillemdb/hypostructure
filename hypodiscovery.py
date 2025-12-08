import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons

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
    def __init__(self, input_dim, latent_dim, max_charts=10, decay=0.9):  # Faster decay to react quickly
        super().__init__()
        self.max_charts = max_charts
        self.decay = decay

        # Track history
        self.register_buffer('avg_usage', torch.ones(max_charts) / max_charts)

        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_charts)
        )
        # Initialize flat to encourage equality start
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.router[-1].bias, 0.0)

        self.charts = nn.ModuleList()
        for _ in range(max_charts):
            expert = nn.Sequential(
                BRSTLinear(input_dim, 128),
                nn.ReLU(),
                BRSTLinear(128, 128),
                nn.ReLU(),
                BRSTLinear(128, latent_dim)
            )
            self.charts.append(expert)

    def forward(self, x):
        logits = self.router(x)
        # Higher temperature to encourage exploration
        weights = nn.functional.gumbel_softmax(logits, tau=2.0, hard=False, dim=1)

        # Update global usage stats (detached from graph)
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

# --- 3. The Natural Loss (With Heating/Cooling Schedule) ---
def natural_loss(z, x, weights, chart_outputs, model, epoch):
    # --- PHASE SCHEDULE ---
    if epoch < 50:
        # HEATING PHASE: Force Exploration
        # Negative coefficient means we MAXIMIZE entropy (reward diversity)
        entropy_coeff = -10.0
    elif epoch < 100:
        # NEUTRAL PHASE: Let geometry take over
        entropy_coeff = 0.0
    else:
        # COOLING PHASE: Force Selection
        # Positive coefficient means we MINIMIZE entropy (punish redundancy)
        # Ramp it up slowly so we don't kill useful charts too fast
        entropy_coeff = min(1.0, 0.01 * (epoch - 100))

    # A. Geometry
    z_prime, _, _ = model(x + torch.randn_like(x) * 0.05)
    loss_inv = nn.functional.mse_loss(z, z_prime)

    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    loss_var = torch.mean(nn.functional.relu(1 - std_z))

    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1]-1, z.shape[1]+1)[:, 1:].flatten()
    loss_cov = off_diag.pow(2).sum()

    # Very high geometry weight to ensure quality
    loss_geo = 100.0*loss_inv + 100.0*loss_var + 5.0*loss_cov

    # B. Parsimony (The Thermostat)
    history = model.avg_usage
    # Entropy of the usage distribution
    global_entropy = -torch.sum(history * torch.log(history + 1e-6))
    metabolic_cost = entropy_coeff * global_entropy

    # C. Local Certainty (Action Gap)
    local_entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-6), dim=1))

    # D. Separation
    loss_sep = torch.tensor(0.0, device=x.device)
    active_indices = [i for i in range(model.max_charts) if history[i] > 0.01]

    if len(active_indices) > 1:
        centers = []
        for i in active_indices:
            w_i = weights[:, i:i+1]
            sum_w = w_i.sum()
            if sum_w > 1e-3:
                center = (chart_outputs[i] * w_i).sum(dim=0) / sum_w
                centers.append(center)

        if len(centers) > 1:
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    d = torch.norm(centers[i] - centers[j])
                    loss_sep += torch.relu(4.0 - d)

    loss_brst = model.compute_brst_loss()

    return loss_geo + metabolic_cost + (5.0 * local_entropy) + (10.0 * loss_sep) + (0.01 * loss_brst)

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
    BATCH_SIZE = 512
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 10 Charts
    model = HypoDiscovery(3, 2, max_charts=10).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)

    epochs = 400  # More epochs for heating/cooling phases

    print(f"Starting Minibatch Discovery...")
    print("Phase 1: Heating (Forced Exploration) | Phase 2: Cooling (Selection)")

    for epoch in range(epochs):
        epoch_loss = 0

        for batch_idx, (x_batch,) in enumerate(dataloader):
            x_batch = x_batch.to(device)

            opt.zero_grad()
            z, w, charts_out = model(x_batch)
            loss = natural_loss(z, x_batch, w, charts_out, model, epoch)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            usage = model.avg_usage.detach().cpu().numpy()
            active = np.sum(usage > 0.02)
            # Print verbose usage to see if they are dying or staying alive
            usage_str = " ".join([f"{u:.2f}" for u in usage])
            print(f"Epoch {epoch}: Avg Loss={epoch_loss/len(dataloader):.2f} | Active={active} | Usage=[{usage_str}]")

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

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_cpu[:,0], X_cpu[:,1], X_cpu[:,2], c=labels_full, cmap='Set1', s=3)
    ax1.set_title("Input")

    ax2 = plt.subplot(1, 2, 2)
    sc = ax2.scatter(z[:,0], z[:,1], c=charts_mapped, cmap='tab20', s=3)
    ax2.set_title(f"Discovered Atlas: {len(u_charts)} Charts")
    plt.colorbar(sc)

    plt.savefig("hypo_minibatch.png", dpi=150)
    print("Saved hypo_minibatch.png")

if __name__ == "__main__":
    run_minibatch()