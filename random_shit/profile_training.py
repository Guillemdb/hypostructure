"""Quick profiling script for HypoPPO v4."""
import cProfile
import pstats
import io
import time

# Run a short training session
import torch
import numpy as np
import gymnasium as gym
from hypoppo_v4_atari import HypoPPOv4Atari

def profile_training():
    print("Setting up environment...")
    num_envs = 8
    
    def make_env():
        return gym.make("CartPole-v1")
    
    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    
    agent = HypoPPOv4Atari(
        input_shape=(4,),
        action_dim=2,
        lr=5e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    obs, _ = envs.reset()
    states = obs
    
    agent.buffer.start_trajectory()
    
    # Time individual components
    times = {
        'action_selection': 0,
        'env_step': 0,
        'buffer_add': 0,
        'update': 0,
        'total': 0,
    }
    
    print("Running profiled training (10 updates)...")
    start_total = time.time()
    
    for update_num in range(10):
        # Collect 128 steps per env
        t0 = time.time()
        for _ in range(128):
            actions = []
            log_probs = []
            values = []
            
            t_action = time.time()
            for i in range(num_envs):
                a, lp, v = agent.select_action(states[i])
                actions.append(a)
                log_probs.append(lp)
                values.append(v)
            times['action_selection'] += time.time() - t_action
            
            t_env = time.time()
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            times['env_step'] += time.time() - t_env
            
            t_buffer = time.time()
            for i in range(num_envs):
                agent.buffer.add_step(states[i], actions[i], rewards[i],
                                     log_probs[i], values[i], dones[i])
                if dones[i]:
                    agent.buffer.start_trajectory()
            times['buffer_add'] += time.time() - t_buffer
            
            states = next_obs
        
        t_update = time.time()
        stats = agent.update(batch_size=64, epochs=3)
        times['update'] += time.time() - t_update
        
        print(f"Update {update_num+1}/10 complete")
    
    times['total'] = time.time() - start_total
    
    envs.close()
    
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (10 updates, 128 steps/env, 8 envs):")
    print("=" * 60)
    
    for name, t in sorted(times.items(), key=lambda x: -x[1]):
        pct = 100 * t / times['total'] if name != 'total' else 100
        print(f"{name:20s}: {t:7.2f}s ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("PER-STEP BREAKDOWN:")
    print("=" * 60)
    total_steps = 10 * 128
    print(f"Action selection:  {1000*times['action_selection']/total_steps:.2f} ms/step")
    print(f"Env step:          {1000*times['env_step']/total_steps:.2f} ms/step")  
    print(f"Buffer add:        {1000*times['buffer_add']/total_steps:.2f} ms/step")
    print(f"Update:            {1000*times['update']/10:.2f} ms/update")

if __name__ == "__main__":
    profile_training()
