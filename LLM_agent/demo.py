from LLM_agent.RGA import RGA
from preschool.text_description.preschool_text import Preschool_Text
import gymnasium as gym
from LLM_agent.utils.rules import set_rules
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
        _, info = env.reset()

        terminated = False
        truncated = False
        env.render()

        while not terminated and not truncated:
            action = agent.take_action(info)
            _, _, terminated, truncated, info = env.step(action)


agent = RGA()
env_id = 'Preschool-v0'
if env_id not in gym.envs.registry:
    gym.register(
        id=env_id,
        entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
        max_episode_steps=100,
    )
env = gym.make("Preschool-v0")
env = Preschool_Text(env)

set_rules(agent)
agent.reasoning_unit.log_reason_theory(logger)

navigate(env, agent)