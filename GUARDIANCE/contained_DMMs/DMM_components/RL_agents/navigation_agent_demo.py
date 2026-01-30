import argparse
from navigation_agent import setup_agent
import torch
import gymnasium as gym
from preschool.grid_world.rand_target import Rand_Target, PrescCoordinates

def visualize(agent, env, seed=None):

        env.set_render_mode('human')
        agent.policy_dqn.eval()  
    

        while True:
            state, _ = env.reset()
            terminated = False     
            truncated = False      

            while(not terminated and not truncated):
                with torch.no_grad():
                    action = agent.policy_dqn(agent.transformation(state)).argmax().item()
                state,reward,terminated,truncated,_ = env.step(("move", action))  
                if reward != 0:
                     pass        

if __name__ == '__main__':
    agent_name = "navigation_agent_700_episodes"
    agent,env= setup_agent(agent_name)
    visualize(agent=agent, env=env)