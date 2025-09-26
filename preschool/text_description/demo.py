#!/usr/bin/env python3
import gymnasium as gym
import argparse
import ast
from preschool.text_description.preschool_text import Preschool_Text
from preschool.grid_world.preschool_grid import Preschool_Grid

"""
Visualizes an instance of the bridge environment with random actions, printing the action, reward, and state at each step.
"""

def demo(env):
    """
    Runs an environment with random actions.
    """
    env.set_render_mode("human")
    env.metadata['render_fps'] = 3

    while True:
        state, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            env.render()
            # randomly choose an action
            action = ("prepare","")
            state, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, State: \n {state}")

if __name__ == '__main__':
    env_id = 'Preschool-v0'
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point='preschool.grid_world.preschool_grid:Preschool_Grid',
            max_episode_steps=100,
        )
    env = gym.make("Preschool-v0")
    env = Preschool_Text(env)
    demo(env)