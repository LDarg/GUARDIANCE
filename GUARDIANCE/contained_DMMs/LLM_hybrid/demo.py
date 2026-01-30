from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.agent_container_PG import Agent_Container_PG
import gymnasium as gym
from preschool.rule_sets.rules import set_rules
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo_LLM.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.DEBUG)


def navigate(env, agent):
    env.set_render_mode("human")
    env.metadata['render_fps'] = 3

    while True:
        rl_obs, info = env.reset()
        agent.update_static_env_info(env.static_facts())

        terminated = False
        truncated = False
        env.render()
        observation = (rl_obs, info)

        while not terminated and not truncated:
            action = agent.take_action(observation)
            rl_obs, _, terminated, truncated, info = env.step(action)
            observation = (rl_obs, info)

# Set up environment and agent
env_id = 'Preschool-v0'
if env_id not in gym.envs.registry:
    gym.register(
        id=env_id,
        entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
        max_episode_steps=100,
    )
env = gym.make("Preschool-v0")

agent = Agent_Container_PG()

# Set initial rules for the agent
set_rules(agent)
agent.moral_module.reasoning_unit.log_reason_theory(logger)

# Start navigation
navigate(env, agent)
