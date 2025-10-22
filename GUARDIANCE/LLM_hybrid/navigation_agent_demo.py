import argparse
from navigation_agent import setup_agent
import torch
import gymnasium as gym
from preschool.grid_world.rand_target import Rand_Target, PrescCoordinates

def visualize(agent, env, seed=None):

        env.set_render_mode('human')
        agent.policy_dqn.eval()  
        
        #state, _ = env.reset()

        while True:
            state, _ = env.reset()
            terminated = False     
            truncated = False      

            while(not terminated and not truncated):
                # select morally permissible action  
                with torch.no_grad():
                    # actions: 0=left,1=down,2=right,3=up
                    action = agent.policy_dqn(agent.transformation(state)).argmax().item()
                state,reward,terminated,truncated,_ = env.step(action)  
                if reward != 0:
                     pass        

if __name__ == '__main__':
    env_id = 'Preschool-v0'
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
            max_episode_steps=100,
        )
    env = gym.make(env_id)
    env = Rand_Target(env)
    env = PrescCoordinates(env)

    agent_name = "navigation_agent_1_episodes"
    agent= setup_agent(agent_name)
    visualize(agent=agent, env=env)