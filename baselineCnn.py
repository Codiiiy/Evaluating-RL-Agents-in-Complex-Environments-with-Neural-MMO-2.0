import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from collections import deque
import random
import argparse
import os

class CNNAgent(nn.Module):
    
    def __init__(self, obs_shape, n_actions):
        super(CNNAgent, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out(obs_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class NMMOBaselineAgent:
    
    def __init__(self, obs_shape, n_actions, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_actions = n_actions
        self.config = config
        
        self.policy_net = CNNAgent(obs_shape, n_actions).to(device)
        self.target_net = CNNAgent(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                     lr=config['learning_rate'])
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = ReplayBuffer(config['buffer_size'])
        
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.max(1)[1].item()
    
    def train_step(self):
        if len(self.memory) < self.config['batch_size']:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['batch_size'])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']


def process_observation(obs_dict):
    if all(isinstance(k, int) for k in obs_dict.keys()):
        feature_arrays = []
        for entity_id, entity_data in obs_dict.items():
            if isinstance(entity_data, dict):
                for feature_name, feature_value in entity_data.items():
                    if isinstance(feature_value, (int, float, np.number)):
                        feature_arrays.append(float(feature_value))
                    elif isinstance(feature_value, np.ndarray):
                        feature_arrays.extend(feature_value.flatten().tolist())
        if feature_arrays:
            obs_array = np.array(feature_arrays, dtype=np.float32)
            size = 15
            if len(obs_array) < size * size:
                padded = np.zeros(size * size, dtype=np.float32)
                padded[:len(obs_array)] = obs_array[:size * size]
                obs_array = padded
            else:
                obs_array = obs_array[:size * size]
            obs_array = obs_array.reshape(size, size)
        else:
            obs_array = np.zeros((15, 15), dtype=np.float32)
    elif 'terrain' in obs_dict:
        obs_array = obs_dict['terrain'].astype(np.float32)
    elif 'tile' in obs_dict:
        obs_array = obs_dict['tile'].astype(np.float32)
    elif 'Tile' in obs_dict:
        obs_array = obs_dict['Tile'].astype(np.float32)
    else:
        obs_list = []
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                obs_list.append(value.flatten())
        if obs_list:
            obs_array = np.concatenate(obs_list)
            size = 15
            if len(obs_array) < size * size:
                padded = np.zeros(size * size, dtype=np.float32)
                padded[:len(obs_array)] = obs_array[:size * size]
                obs_array = padded
            else:
                obs_array = obs_array[:size * size]
            obs_array = obs_array.reshape(size, size)
        else:
            obs_array = np.zeros((15, 15), dtype=np.float32)
    
    if len(obs_array.shape) == 3:
        obs_array = np.transpose(obs_array, (2, 0, 1))
    elif len(obs_array.shape) == 2:
        obs_array = np.expand_dims(obs_array, axis=0)
    
    return obs_array.astype(np.float32)


def train_agent(env, agent, args, use_wandb=False):
    for episode in range(args.episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0] if len(reset_result) > 0 else {}
        else:
            obs = reset_result
        
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        
        prev_obs = {}
        prev_actions = {}
        
        for agent_id in env.agents:
            if agent_id in obs:
                prev_obs[agent_id] = process_observation(obs[agent_id])
        
        while env.agents and episode_steps < args.max_steps:
            actions = {}
            
            for agent_id in env.agents:
                if agent_id in prev_obs:
                    state = prev_obs[agent_id]
                    action_idx = agent.select_action(state, training=True)
                    prev_actions[agent_id] = action_idx
                    actions[agent_id] = env.action_space(agent_id).sample()
            
            try:
                step_result = env.step(actions)
                
                if isinstance(step_result, tuple) and len(step_result) >= 4:
                    next_obs, rewards, dones, truncated = step_result[:4]
                else:
                    break
                
                if args.render:
                    env.render()
                
                for agent_id in list(prev_obs.keys()):
                    if agent_id in next_obs:
                        state = prev_obs[agent_id]
                        next_state = process_observation(next_obs[agent_id])
                        reward = rewards.get(agent_id, 0)
                        done = dones.get(agent_id, False) or truncated.get(agent_id, False)
                        action = prev_actions.get(agent_id, 0)
                        
                        agent.memory.push(state, action, reward, next_state, done)
                        episode_reward += reward
                
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                if agent.steps % agent.config['target_update'] == 0:
                    agent.update_target_network()
                
                agent.steps += 1
                episode_steps += 1
                
                prev_obs = {}
                for agent_id in env.agents:
                    if agent_id in next_obs:
                        prev_obs[agent_id] = process_observation(next_obs[agent_id])
                
            except Exception as e:
                print(f"Error during step: {e}")
                import traceback
                traceback.print_exc()
                break
        
        agent.update_epsilon()
        agent.episodes += 1
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Steps: {episode_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory)}")
        
        if use_wandb:
            wandb.log({
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "avg_loss": avg_loss,
                "epsilon": agent.epsilon,
                "memory_size": len(agent.memory)
            })
        
        if (episode + 1) % args.save_interval == 0:
            save_path = f"{args.checkpoint_dir}/checkpoint_ep{episode + 1}.pt"
            agent.save(save_path)
            print(f"Saved checkpoint to {save_path}")
            if use_wandb:
                wandb.save(save_path)
    
    final_path = f"{args.checkpoint_dir}/final_checkpoint.pt"
    agent.save(final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    if use_wandb:
        wandb.save(final_path)


if __name__ == "__main__":
    import nmmo
    
    parser = argparse.ArgumentParser(description="Train NMMO DQN Agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="NMMO")
    parser.add_argument("--wandb-entity", type=str, default="NeuralMMOUTRGV")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    config = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "target_update": args.target_update
    }
    
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)
    
    try:
        print("Initializing Neural MMO environment...")
        env = nmmo.Env()
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0] if len(reset_result) > 0 else {}
        else:
            obs = reset_result
        
        print(f"Number of agents: {len(env.agents)}")
        
        first_agent = list(env.agents)[0]
        agent_obs = obs[first_agent]
        processed_obs = process_observation(agent_obs)
        obs_shape = processed_obs.shape
        print(f"Observation shape: {obs_shape}")
        
        action_space = env.action_space(first_agent)
        if hasattr(action_space, 'n'):
            n_actions = action_space.n
        elif hasattr(action_space, 'nvec'):
            n_actions = int(np.prod(action_space.nvec))
        else:
            n_actions = 256
        print(f"Action space size: {n_actions}")
        
        print("\nInitializing agent...")
        agent = NMMOBaselineAgent(obs_shape, n_actions, config)
        
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            agent.load(args.checkpoint)
        
        print(f"Device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
        
        print("\nStarting training...")
        train_agent(env, agent, args, use_wandb=args.use_wandb)
        
        if args.use_wandb:
            wandb.finish()
        
        if args.render:
            env.close()
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if args.use_wandb:
            wandb.finish()
