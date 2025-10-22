import argparse
import navigation_agent as navigation_agent
import torch

def visualize(agent, env, seed=None):

        env.set_render_mode('human')
        agent.policy_dqn.eval()  
        
        state, _ = env.reset()

        while True:
            terminated = False     
            truncated = False       
            #obeservation = env.observation()   

            obs_dict= training_env.get_obs_dict()
            while(not terminated and not truncated):
                obs = env.get_obs_dict() 
                # select morally permissible action  
                with torch.no_grad():
                    #take an action that is conform with the agent's moral obligations according to its current reason theory
                    action_preferences = [tensor.item() for tensor in agent.policy_dqn(agent.transformation(state))]
                    action = action_preferences.index(max(action_preferences))  
                state,_,terminated,truncated,_ = env.step(action)          

if __name__ == '__main__':
    navigation_agent, training_env = navigation_agent.setup_agent("agent")
    visualize(agent=navigation_agent, env=training_env)