from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.agent_container_PG import Agent_Container_PG
import gymnasium as gym
from preschool.rule_sets.rules import set_rules
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo_LLM.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.DEBUG)

logging.info("Starting LLM_hybrid agent demo")

def navigate(env, agent):
    env.set_render_mode("human")
    env.metadata['render_fps'] = 3

    while True:
        rl_obs, info = env.reset()
        agent.static_env_info = env.static_facts()

        terminated = False
        truncated = False
        env.render()
        observation = (rl_obs, info)

        while not terminated and not truncated:
            action = agent.take_action(observation)
            rl_obs, _, terminated, truncated, info = env.step(action)
            observation = (rl_obs, info)

# set up environment and agent
agent = Agent_Container_PG()
env_id = 'Preschool-v0'
if env_id not in gym.envs.registry:
    gym.register(
        id=env_id,
        entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
        max_episode_steps=100,
    )
env = gym.make("Preschool-v0")

# set initial rules for the agent
set_rules(agent)
agent.reasoning_unit.log_reason_theory(logger)

# start navigation
navigate(env, agent)
