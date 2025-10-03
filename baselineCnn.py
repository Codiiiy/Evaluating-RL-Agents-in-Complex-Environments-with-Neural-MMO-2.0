import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from collections import deque
import random
import argparse

wandb.init(project="NMMO", entity="NeuralMMOUTRGV", config={
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 50000,
    "batch_size": 64,
    "target_update": 10
})

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
    
    def __init__(self, obs_shape, n_actions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_actions = n_actions
        
        self.policy_net = CNNAgent(obs_shape, n_actions).to(device)
        self.target_net = CNNAgent(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                     lr=wandb.config.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = ReplayBuffer(wandb.config.buffer_size)
        
        self.epsilon = wandb.config.epsilon_start
        self.epsilon_min = wandb.config.epsilon_end
        self.epsilon_decay = wandb.config.epsilon_decay
        
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
        if len(self.memory) < wandb.config.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(
            wandb.config.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * wandb.config.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        wandb.log({
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_value": current_q.mean().item(),
            "step": self.steps
        })
        
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
        wandb.save(path)
    
    def load(self, path):
        checkpoint = torch.load(path)
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


def train_agent(env, agent, n_episodes=1000):
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0] if len(reset_result) > 0 else {}
        else:
            obs = reset_result
        
        episode_reward = 0
        episode_steps = 0
        prev_obs = {}
        for agent_id in env.agents:
            if agent_id in obs:
                prev_obs[agent_id] = process_observation(obs[agent_id])
        
        while env.agents and episode_steps < 1000:
            actions = {}
            action_indices = {}
            for agent_id in env.agents:
                if agent_id in prev_obs:
                    state = prev_obs[agent_id]
                    action_idx = agent.select_action(state)
                    action_indices[agent_id] = action_idx
                    actions[agent_id] = env.action_space(agent_id).sample()
            try:
                step_result = env.step(actions)
            except Exception as e:
                print(f"Error during step: {e}")
                import traceback
                traceback.print_exc()
                break
            if isinstance(step_result, tuple):
                if len(step_result) == 4:
                    obs, rewards, dones, infos = step_result
                elif len(step_result) == 5:
                    obs, rewards, dones, truncated, infos = step_result
                else:
                    print(f"Unexpected step result: {len(step_result)} elements")
                    break
            else:
                print(f"Step returned non-tuple: {type(step_result)}")
                break
            curr_obs = {}
            for agent_id in env.agents:
                if agent_id in obs:
                    curr_obs[agent_id] = process_observation(obs[agent_id])
            for agent_id in action_indices.keys():
                if agent_id in rewards:
                    reward = rewards[agent_id]
                    done = dones.get(agent_id, False)
                    if agent_id in curr_obs:
                        next_state = curr_obs[agent_id]
                    else:
                        next_state = np.zeros_like(prev_obs[agent_id])
                    agent.memory.push(
                        prev_obs[agent_id],
                        action_indices[agent_id],
                        reward,
                        next_state,
                        done
                    )
                    episode_reward += reward
            loss = agent.train_step()
            prev_obs = curr_obs
            episode_steps += 1
            agent.steps += 1
            if agent.steps % wandb.config.target_update == 0:
                agent.update_target_network()
        agent.update_epsilon()
        agent.episodes += 1
        wandb.log({
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "episode": episode
        })
        if episode % 100 == 0:
            agent.save(f"checkpoint_ep{episode}.pt")
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Steps={episode_steps}, Epsilon={agent.epsilon:.3f}")
    wandb.finish()


if __name__ == "__main__":
    import nmmo
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()

    try:
        print("Initializing Neural MMO environment...")
        env = nmmo.Env()
        reset_result = env.reset()
        print(f"Reset returned: {type(reset_result)}")
        if isinstance(reset_result, tuple):
            print(f"  Tuple length: {len(reset_result)}")
            for i, item in enumerate(reset_result):
                print(f"  Element {i}: {type(item)}")
            obs = reset_result[0] if len(reset_result) > 0 else {}
        else:
            obs = reset_result
        print(f"\nObs type: {type(obs)}")
        print(f"Number of agents: {len(env.agents)}")
        print(f"Agent IDs: {list(env.agents)[:5]}...")
        if isinstance(obs, dict):
            print(f"Obs keys: {list(obs.keys())[:5]}...")
        first_agent = env.agents[0]
        agent_obs = obs[first_agent]
        print(f"\nAgent {first_agent} observation structure:")
        print(f"  Type: {type(agent_obs)}")
        print(f"  Keys: {list(agent_obs.keys())[:10]}..." if isinstance(agent_obs, dict) else "Not a dict")
        if isinstance(agent_obs, dict) and len(agent_obs) > 0:
            first_key = list(agent_obs.keys())[0]
            first_value = agent_obs[first_key]
            print(f"\n  First key ({first_key}) contains: {type(first_value)}")
            if isinstance(first_value, dict):
                print(f"    Sub-keys: {list(first_value.keys())}")
                if len(first_value) > 0:
                    sample_key = list(first_value.keys())[0]
                    sample_value = first_value[sample_key]
                    print(f"    Sample ({sample_key}): {type(sample_value)}, value: {sample_value}")
        processed_obs = process_observation(agent_obs)
        obs_shape = processed_obs.shape
        print(f"\nProcessed observation shape: {obs_shape}")
        action_space = env.action_space(first_agent)
        if hasattr(action_space, 'n'):
            n_actions = action_space.n
        elif hasattr(action_space, 'nvec'):
            n_actions = int(np.prod(action_space.nvec))
        else:
            n_actions = 25
        print(f"Action space size: {n_actions}")
        print(f"Action space type: {type(action_space)}")
        print("\nInitializing CNN agent...")
        agent = NMMOBaselineAgent(obs_shape, n_actions)
        print(f"Device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
        print("\nStarting training...")
        train_agent(env, agent, n_episodes=args.episodes)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
