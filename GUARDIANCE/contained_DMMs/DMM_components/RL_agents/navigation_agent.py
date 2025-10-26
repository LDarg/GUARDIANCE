import torch
import random
from tqdm import tqdm
from GUARDIANCE.contained_DMMs.DMM_components.RL_agents.utils import drl
from GUARDIANCE.contained_DMMs.DMM_components.RL_agents.utils.plotting import plot_training_progress
import numpy as np
import logging
from torch import nn
import os
import gymnasium as gym
from preschool.grid_world.rand_target import Rand_Target, PrescCoordinates, Extended
import sys

logger = logging.getLogger(__name__)

def get_agent_info(agent_name):
    dir_name = get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def load_modules(agent, agent_info):
    agent.policy_dqn.load_state_dict(agent_info["policy_state_dict"])
    agent.target_dqn.load_state_dict(agent_info["target_state_dict"])

def setup_agent(agent_name):
    agent_info = get_agent_info(agent_name)
    env = agent_info["env"]
    agent = RL_agent(env)
    load_modules(agent, agent_info)
    return (agent, env)

def get_dir_name():
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(current_file_dir, 'trained_models')

def save_agent(agent, agent_name):
    
    save_dict = {
        "policy_state_dict": agent.policy_dqn.state_dict(),
        "target_state_dict": agent.target_dqn.state_dict(),
        "env": agent.env
    }
    dir_name = get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")
    torch.save(save_dict, file_path)
    logger.info(f"Agent saved to {file_path}")

class RL_agent:
    def __init__(self, env, lr = 0.001, sync_rate = 1000, replay_memory_size=1000, mini_batch_size= 32):

        self.env = env

        #set up hyperparameters for DRL
        self.lr = lr        
        self.discount = 0.9            
        self.sr = sync_rate       
        self.replay_memory_size = replay_memory_size     
        self.mini_batch_size = mini_batch_size          

        self.loss_fn = nn.MSELoss()          
        self.optimizer = None                

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        #set up policy and target network
        self.policy_dqn = drl.DQN_2hiddenlayers(in_states=self.num_inputs, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        self.target_dqn = drl.DQN_2hiddenlayers(in_states=self.num_inputs, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        
        self.logger = logging.getLogger(__name__)

    def train(self, episodes, agent_name, env=None, pb=True, seed=None):

        if not env:
            env = self.env
        
        epsilon = 1 # 0: no randomness; 1:completely random
        k = 3 # decay rate for epsilon
        memory = drl.ReplayMemory(self.replay_memory_size)

        # initially sync the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.lr)

        rewards_per_episode = np.zeros(episodes)

        # count steps until policy and target network are synced
        step_count=0

        iterator = tqdm(range(episodes), desc="Training Instr") if pb else range(episodes)

        for i in iterator:
            state, info = env.reset(seed =seed) 
            terminated = False      
            truncated = False      
            steps_in_episode = 0

            while(not terminated and not truncated):

                # select action epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up, 4:help child
                else:
                    # select best action            
                    with torch.no_grad():
                        action = self.policy_dqn(self.transformation(state)).argmax().item()

                # execute the agent's action choice
                new_state,reward,terminated,truncated, info = env.step(("move", action))


                #self.HER_transformation(new_state,reward,terminated,truncated, info)
                # add sample to memory
                memory.append((state, action, new_state, reward, terminated, truncated)) 

                # update the state
                state = new_state

                # increment counter for synching the target network with the policy network
                step_count+=1

                if reward != 0:
                    pass

                # update the policy network
                if len(memory)>self.mini_batch_size: 
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch) 
                
                # sync policy and target network
                if step_count > self.sr:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0

            if reward != 0:
                rewards_per_episode[i] = reward

            # reduce chance to randomly pick an action
            epsilon=np.exp(-k * (i/episodes))

        env.close()

        #sum_rewards = np.zeros(episodes)
        sum_rewards = np.array([np.sum(rewards_per_episode[max(0, x-200):x+1]) for x in range(episodes)])

        plot_training_progress(sum_rewards, agent_name)


    def optimize(self, mini_batch):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated, truncated in mini_batch:

            if terminated or truncated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount * self.target_dqn(self.transformation(new_state)).max()
                    )

            current_q = self.policy_dqn(self.transformation(state))
            current_q_list.append(current_q)

            target_q = self.target_dqn(self.transformation(state)) 
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transformation(self, observation):
        return  torch.tensor(observation, dtype=torch.float32)


    def train_navigation_policy(self, env, episodes, name, pb=True):
        if not env:
            env = self.env
        self.train(episodes, name, env, pb=pb)
        save_agent(self, name)

    def land_tiles_next_to_person(env, coordinates):
        directions = env.directions
        land_tiles_next_to_person = []
        for direction in directions:
                neighbor = tuple(tuple(coordinates)  + direction)
                if 0  < neighbor[0] < env.bridge_map.width and  0 < neighbor[1] < env.bridge_map.height:
                    if env.bridge_map.get_grid_type(neighbor) != env.bridge_map.grid_types["water"]:
                        land_tiles_next_to_person.append(neighbor)
        return land_tiles_next_to_person
     

if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in root_logger.handlers):
        root_logger.addHandler(stream_handler)

    logger.info("Logger configured to output to terminal")

    env_id = 'Preschool-v0'

    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
            max_episode_steps=100,
        )
    env = gym.make(env_id)
    env = Rand_Target(env)
    env = Extended(env)
    agent = RL_agent(env)

    training_episodes = 800
    train_navigation_policy = agent.train_navigation_policy(env=env, episodes=training_episodes, name=f"navigation_agent_{training_episodes}_episodes", pb=True)


