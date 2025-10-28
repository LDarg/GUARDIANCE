from GUARDIANCE.contained_DMMs.LLM_hybrid.contained_LLM_hybrid import contained_LLM_PG
from preschool.grid_world.preschool_grid import Preschool_Grid
import gymnasium as gym
from preschool.rule_sets.rules import set_rules
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo_LLM.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.DEBUG)

logging.info("Starting LLM agent demo")

def navigate(env, agent):
    env.set_render_mode("human")
    env.metadata['render_fps'] = 3

    while True:
        rl_obs, info = env.reset()
        agent.static_env_info = env.static_facts()

        terminated = False
        truncated = False
        env.render()

        while not terminated and not truncated:
            action = agent.take_action(rl_obs, info)
            rl_obs, _, terminated, truncated, info = env.step(action)

# set up environment and agent
agent = contained_LLM_PG()
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