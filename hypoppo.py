import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

try:
    import plangym
except (ImportError, ModuleNotFoundError, Exception) as e:
    plangym = None

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, AtariPreprocessing

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, next_state, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []

    def get_batch(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.log_probs),
            np.array(self.dones)
        )

# Nature CNN for Atari (Mnih et al. 2015)
class NatureCNN(nn.Module):
    def __init__(self, input_channels, feature_dim=512):
        super(NatureCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, state_val, action_dim, hidden_dim=64, continuous=True):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        self.is_cnn = False
        
        if isinstance(state_val, (tuple, list)) and len(state_val) == 3:
            self.is_cnn = True
            h, w, c = state_val
            if c < h and c < w:
                input_channels = c
            else:
                input_channels = state_val[0] 
                
            self.trunk = NatureCNN(input_channels, feature_dim=512)
            trunk_output_dim = 512
        else:
            input_dim = state_val if isinstance(state_val, int) else np.prod(state_val)
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            trunk_output_dim = hidden_dim
        
        self.critic = nn.Linear(trunk_output_dim, 1)

        if self.continuous:
            self.actor_mean = nn.Linear(trunk_output_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor_logits = nn.Linear(trunk_output_dim, action_dim)

    def forward(self, state):
        if self.is_cnn and state.max() > 1.1:
            state = state / 255.0
            
        features = self.trunk(state)
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor_logits(features)
            dist = torch.distributions.Categorical(logits=logits)
        
        return dist, value

class HypoPPO:
    def __init__(self, state_shape, action_dim, 
                 continuous=True,
                 lr=3e-4, 
                 gamma=0.99,
                 kl_target=None,      # Disabled by default (Use Clipping)
                 entropy_target=None, # Disabled by default (Use Fixed Coeff)
                 entropy_coef=0.01,   # Fixed Entropy Coeff
                 rho_lr=0.01,         
                 scale_reg_coeff=0.0, # Disabled by default
                 scale_gamma=2.0,      
                 device='cpu'):
        
        self.device = device
        self.continuous = continuous
        self.state_shape = state_shape
        self.gamma = gamma
        self.model = ActorCritic(state_shape, action_dim, continuous=continuous).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.buffer = RolloutBuffer()
        
        self.rho = 0.0 
        self.rho_lr = rho_lr
        self.kl_target = kl_target
        self.beta_kl = 1.0 
        self.target_entropy = entropy_target
        self.entropy_coef = entropy_coef
        
        # Only init alpha if dynamic entropy is used
        if self.target_entropy is not None:
             self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
             self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
             self.log_alpha = None
             self.alpha_optimizer = None
             
        self.scale_reg_coeff = scale_reg_coeff
        self.scale_gamma = scale_gamma

    def _to_tensor(self, state):
        if hasattr(state, "__array__"):
             state = np.array(state)
        tens = torch.FloatTensor(state).to(self.device)
        if self.model.is_cnn:
             if tens.dim() == 3:
                 if tens.shape[2] in [1, 3, 4] and tens.shape[0] > 4:
                     tens = tens.permute(2, 0, 1)
             elif tens.dim() == 4:
                 if tens.shape[3] in [1, 3, 4] and tens.shape[1] > 4:
                     tens = tens.permute(0, 3, 1, 2)
        return tens

    def select_action(self, state):
        state_tens = self._to_tensor(state)
        if state_tens.dim() == 3 and self.model.is_cnn:
             state_tens = state_tens.unsqueeze(0)
        elif state_tens.dim() == 1 and not self.model.is_cnn:
             state_tens = state_tens.unsqueeze(0)
            
        with torch.no_grad():
            dist, value = self.model(state_tens)
            action = dist.sample()
            
            if self.continuous:
                log_prob = dist.log_prob(action).sum(dim=-1)
                action_np = action.cpu().numpy()[0]
            else:
                log_prob = dist.log_prob(action)
                action_np = action.cpu().numpy()[0] 
        
        return action_np, log_prob.cpu().numpy()[0], value.item()

    def update_rho(self, rewards):
        batch_mean = rewards.mean().item()
        self.rho = self.rho + self.rho_lr * (batch_mean - self.rho)
        return self.rho

    def update(self, batch_size=64, epochs=4):
        b_states, b_actions, b_rewards, b_next_states, b_log_probs, b_dones = self.buffer.get_batch()
        if len(b_rewards) == 0:
             return {}

        states = self._to_tensor(b_states)
        next_states = self._to_tensor(b_next_states)
        actions = torch.tensor(b_actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(b_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor(b_log_probs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(b_dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        n_samples = states.size(0)
        
        with torch.no_grad():
            _, next_values = self.model(next_states)
            _, curr_values = self.model(states)
            
            if self.gamma < 1.0:
                 target_values = rewards + self.gamma * next_values * (1 - dones)
            else:
                 target_values = rewards - self.rho + next_values * (1 - dones)
            
            advantages = target_values - curr_values
            # Global Advantage Normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.update_rho(rewards)
        history = {'loss_p': [], 'loss_v': [], 'loss_reg': [], 'loss_total': [], 'kl': [], 'rho': []}
        
        for _ in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_targets = target_values[idx]
                mb_adv = advantages[idx]
                mb_old_log_probs = old_log_probs[idx]
                
                dist, values = self.model(mb_states)
                if self.continuous:
                    log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    act = mb_actions.squeeze(-1) if mb_actions.dim() > 1 else mb_actions
                    log_probs = dist.log_prob(act)
                    entropy = dist.entropy().mean()
                
                ratios = torch.exp(log_probs - mb_old_log_probs)
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - log_probs).mean()
                
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * mb_adv
                loss_policy_clip = -torch.min(surr1, surr2).mean()
                
                # Check for None explicitly to avoid Tensor vs None errors
                if self.kl_target is not None:
                    loss_policy_kl = self.beta_kl * approx_kl
                else:
                    loss_policy_kl = torch.tensor(0.0).to(self.device)
                    
                total_policy_loss = loss_policy_clip + loss_policy_kl
                loss_value = F.mse_loss(values, mb_targets)
                
                if self.target_entropy is not None:
                     alpha = self.log_alpha.exp().item()
                     loss_entropy = - alpha * entropy
                     loss_alpha = self.log_alpha * (entropy.detach() - self.target_entropy)
                else:
                     loss_entropy = - self.entropy_coef * entropy
                     loss_alpha = torch.tensor(0.0).to(self.device)
                
                if self.scale_reg_coeff > 0.0:
                    trunk_params = list(self.model.trunk.parameters())
                    grads_v = torch.autograd.grad(loss_value, trunk_params, create_graph=True, retain_graph=True)
                    grads_p = torch.autograd.grad(total_policy_loss, trunk_params, create_graph=True, retain_graph=True)
                    norm_v = torch.sqrt(sum([torch.sum(g**2) for g in grads_v]))
                    norm_p = torch.sqrt(sum([torch.sum(g**2) for g in grads_p]) + 1e-6)
                    scale_ratio = norm_v / norm_p
                    reg_term = torch.clamp(scale_ratio - self.scale_gamma, min=0.0)**2
                    reg_term = torch.clamp(reg_term, max=100.0)
                    loss_reg = self.scale_reg_coeff * reg_term
                else:
                    loss_reg = torch.tensor(0.0).to(self.device)
                
                total_loss = total_policy_loss + 0.5 * loss_value + loss_entropy + loss_reg
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                if self.target_entropy is not None:
                    self.alpha_optimizer.zero_grad()
                    loss_alpha.backward()
                    self.alpha_optimizer.step()
                
                history['loss_p'].append(total_policy_loss.item())
                history['loss_v'].append(loss_value.item())
                history['loss_reg'].append(loss_reg.item())
                history['loss_total'].append(total_loss.item())
                history['kl'].append(approx_kl.item())
            history['rho'].append(self.rho)
        self.buffer.clear()
        
        if self.kl_target is not None:
            mean_kl = np.mean(history['kl'])
            if mean_kl > self.kl_target * 1.5:
                self.beta_kl *= 2.0
            elif mean_kl < self.kl_target / 1.5:
                self.beta_kl *= 0.5
            self.beta_kl = np.clip(self.beta_kl, 0.01, 100.0)
            history['beta'] = self.beta_kl 
            
        return {k: np.mean(v) if isinstance(v, list) else v for k, v in history.items()}

def evaluate(agent, env_name):
    try:
        if "NoFrameskip" in env_name:
             import ale_py
             gym.register_envs(ale_py)
             env = gym.make(env_name, render_mode=None)
             env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, grayscale_newaxis=True, scale_obs=False)
        else:
             env = gym.make(env_name)
    except:
        return 0.0
    state, info = env.reset()
    if hasattr(state, "__array__"): state = np.array(state)
    total_reward = 0
    done = False
    trunc = False
    while not (done or trunc):
        action, _, _ = agent.select_action(state)
        state, reward, done, trunc, _ = env.step(action)
        if hasattr(state, "__array__"): state = np.array(state)
        total_reward += reward
    env.close()
    return total_reward

if __name__ == "__main__":
    print("HypoPPO CartPole Stable (Robust)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    env_name = "CartPole-v1"

    try:
        if "NoFrameskip" in env_name:
             import ale_py
             gym.register_envs(ale_py)
             env = gym.make(env_name, render_mode=None)
             env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, grayscale_newaxis=True, scale_obs=False)
        else:
             env = gym.make(env_name)
    except Exception as e:
        print(f"Env create failed: {e}")
        env = gym.make("CartPole-v1")
    
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"Task: {env_name}, State: {state_shape}, Action: {action_dim}")
    
    # Stable PPO Mode
    agent = HypoPPO(state_shape, action_dim, continuous=False,
                    lr=5e-4,             # Lower/Mid LR for stability
                    gamma=0.99,
                    kl_target=None,
                    entropy_target=None,
                    entropy_coef=0.01,   # Standard small entropy
                    scale_reg_coeff=0.0,
                    device=device) 
    
    print("\nStarting Training Loop...")
    steps = 0
    max_steps = 50000 
    update_interval = 2048 # Standard batch size
    
    state, info = env.reset()
    while steps < max_steps:
        action, log_prob, val = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer.add(state, action, reward, next_state, log_prob, done)
        state = next_state
        steps += 1
        if done:
            state, info = env.reset()
        if steps % update_interval == 0:
            stats = agent.update(batch_size=64, epochs=4) # Reduce Epochs to 4
            score = evaluate(agent, env_name)
            print(f"Step {steps}: Score={score:.1f}, LTot={stats['loss_total']:.3f}, P={stats['loss_p']:.3f}, V={stats['loss_v']:.3f}, Reg={stats['loss_reg']:.4f}, Rho={stats['rho']:.3f}")

    print("Finished.")
