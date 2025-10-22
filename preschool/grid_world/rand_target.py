import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from random import random


class Rand_Target(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)

            self.render_mode_setting = self.env.render_mode

            self.target_position = np.array([0,0])
            len_obs_dict = len(self.get_obs_dict())
            observation_size = len_obs_dict * self.env.total_grid_cells
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(observation_size,),
                dtype=np.float32
            )

            #modify this to as a hyperparameter for GCRL with HER
            self.max_steps = 20

        def observation(self):
            obs_dict = self.get_obs_dict()

            # flatten 
            agent_flat = obs_dict["agent_window"].flatten()
            target_flat = obs_dict["target_window"].flatten()

            # concatenate into one 1D array
            nn_input = np.concatenate([agent_flat, target_flat])

            return np.array(nn_input.astype(np.float32))
    
        def get_obs_dict(self):

            # agent's position (one-hot encoding)
            agent_window = np.zeros((self.env.grid_height, self.env.grid_width))
            agent_x, agent_y = self.agent_coordinates
            agent_window[agent_y, agent_x] = 1

            # target position (one-hot encoding)
            target_window = np.zeros((self.env.grid_height, self.env.grid_width))
            target_x, target_y = self.target_position
            target_window[target_y, target_x] = 1

            observation = {
                "agent_window": agent_window,
                "target_window": target_window
            }
            return observation
        
        def reset(self, seed=None, options=None):
            self.target_position = self.env.map.random_location()
            observation, info= super().reset(seed=seed, options=options)
            self.render_mode_setting = self.env.render_mode
            self.render()
            self.steps_counter = 0
            observation = self.observation()
            return observation, info
        
        def set_render_mode(self, render_mode):
            self.render_mode_setting = render_mode
            self.env.set_render_mode(render_mode)

        def render(self):
             if self.env.render_mode == "human":
                canvas = self.env.render_frame()
                canvas = self.render_target_position(canvas)
                self.env.handle_events()
                self.window.blit(canvas, (0, 0))
                self.env.update_display()

        
        def step(self, action):
            self.env.set_render_mode(None)
            observation,reward,terminated,truncated,info = super().step(action)
            self.env.set_render_mode(self.render_mode_setting)
            self.render()
            if np.equal(self.env.agent_coordinates, self.target_position).all():
                reward = 100
            self.steps_counter += 1
            if self.steps_counter >= self.max_steps:
                truncated = True
            observation = self.observation()
            info = self.get_obs_dict()
            return observation,reward,terminated,truncated,info

        def render_target_position(self, canvas):
            pix_width_size = self.env.navigation_area_width / self.env.map.width
            pix_height_size = self.env.navigation_area_height / self.env.map.height
            target_pos = ((self.target_position + 0.5) * np.array([pix_width_size, pix_height_size])).astype(int)
            pygame.draw.rect(
                canvas,
                (0, 100, 0),
                pygame.Rect(
                    (target_pos[0] - int(pix_width_size / 6), target_pos[1] - int(pix_height_size / 6)),
                    (int(pix_width_size / 3), int(pix_height_size / 3)),
                ),
            )
            return canvas