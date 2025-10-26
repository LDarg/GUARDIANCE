import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import random
import os
from preschool.resource_manager import ResourceManager
import copy

pygame.init()
resource_manager = ResourceManager() 
resource_manager.load_icons()

"""
A wrapper for the preschool environment that abstracts away low-level details of the environment.
It simplifies navigation by reducing the grid world to just two zones, A and B.
The wrapper also provides a text-based observation interface.
Designed for demonstrating a PoC of the governance architecture with an LLM as DMM.
"""
class Preschool_Text(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)

        def render(self):
            if self.render_mode == "human":
                return self.render_frame()
            
        def reset(self, seed=None, options=None):
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            super().reset(seed=seed)
            self.initialize_at_reset(self.np_random)
            observation = self.observation()
            info = self.get_facts()

            self.window = None
            self.render()

            return observation, info
        
        def get_facts(self):
            children_facts = set()
            zones_facts = set()
            agent_zone = self.map.get_zone(self.agent_coordinates).id

            for child in self.map.children:
                children_facts.add((child.id, f"{child.condition}", self.map.get_zone(child.coordinates).name, self.map.get_zone(child.coordinates).id))

            for zone in self.map.zones:
                if zone.happening is not None:
                    zones_facts.add((zone.id, zone.happening, zone.name))

            stations_zones = [{"zone_id": str(self.map.get_zone(learning_station.coordinates).id)} for learning_station in self.map.learning_stations]
            
            return {
                "children": children_facts,
                "zones": zones_facts,
                "agent_zone": agent_zone,
                "stations_zones": stations_zones,
                "zone_ids": [zone.id for zone in self.map.zones]
            }

        def step(self, action):
            reward = 0
            terminated = False

            # execute the action if it is moving between zones
            if action[0] == "move":
                try:
                    zone_name = next(zone.name for zone in self.map.zones if zone.id == action[1])
                except StopIteration:
                    raise ValueError(f"No zone found with id: {action[1]}")
                if zone_name == "A":
                    self.agent_coordinates = np.array([0,0])
                elif zone_name == "B":
                    self.agent_coordinates = np.array([7,7])

            # exceute the action if it is preparing a learning station
            if action[0] == "prepare":
                for learning_station in self.map.learning_stations:
                    if self.map.get_zone(self.agent_coordinates).name == self.map.get_zone(learning_station.coordinates).name:
                        learning_station.progress()
                        if learning_station.finished():
                            self.map.learning_stations.remove(learning_station)
                        break

            # execute the action if it is helping a child
            if action[0] == "help":
                for child in self.map.children:
                    #TODO: check if agent in the same zone as the child
                    if child.id == action[1]:
                        if self.map.config.resolutions[child.condition].replace(" ", "_") == action[2]:
                            self.map.children.remove(child)
                            break

            # generate moral goals and happenings with a certain probability
            # SIMPLIFICATION: ensure that for every condition, there is only one instance of a child with that condition
            if random.random() < 1:
                if len(self.map.children) < len(self.map.config.conditions):
                    self.map.generate_moral_goal_unique()
            if random.random() < 0.3:
                self.map.generate_happening()

            observation = self.observation()
            info = self.get_facts()

            self.render()
        
            return observation, reward, terminated, False, info

        def render_frame(self):
            facts_area_height = 300
            zone_name_area_height = 40
            if self.window is None:
                pygame.font.init()
                pygame.display.init()
                pygame.display.set_caption("Preschool")
                self.window = pygame.display.set_mode(
                    (self.navigation_area_width, self.navigation_area_height+facts_area_height+zone_name_area_height)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.fill((0, 0, 0))

            # Create canvas on which the grid is drawn
            canvas = pygame.Surface((self.navigation_area_width, self.navigation_area_height))
            canvas.fill((0, 0, 255))  # fallback background

            pix_width_size = self.navigation_area_width / self.map.width
            pix_height_size = self.navigation_area_height / self.map.height

            # Drawing area for the zone names
            pygame.draw.rect(self.window, (50, 50, 80), (0, 0, self.navigation_area_width, zone_name_area_height))
            x_offset = self.navigation_area_width // ((len(self.map.zones)*2))
            y_offset = 5 

            # Draw zones
            font_zone_name = pygame.font.Font(None, 28)
            sorted_zones = sorted(self.map.zones, key=lambda zone: zone.name)
            for zone in sorted_zones:
                zone_coords = np.array(zone.coordinates)

                # Bounding box of the zone
                min_x, min_y = np.min(zone_coords, axis=0)
                max_x, max_y = np.max(zone_coords, axis=0)

                rect_x = min_x * pix_width_size
                rect_y = min_y * pix_height_size
                rect_w = (max_x - min_x + 1) * pix_width_size
                rect_h = (max_y - min_y + 1) * pix_height_size

                # Set colors for zones 
                if zone.name == "A":
                    color = (255, 255, 197)
                elif zone.name == "B":
                    color = (173, 216, 229)

                if zone.happening:
                    color = (255,0,0)

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(rect_x, rect_y, rect_w, rect_h),
                )

                # Draw zone name 
                text_surface = font_zone_name.render(zone.name, True, (255, 255, 255))
                self.window.blit(text_surface, (x_offset, y_offset))
                x_offset += x_offset + self.navigation_area_width //4

                icon_w, icon_h = resource_manager.book_icon.get_width(), resource_manager.book_icon.get_height()
                icon_x = rect_x
                icon_y = rect_y + rect_h - icon_h

                for learning_station in self.map.learning_stations:
                    if self.map.coordinates_in_zone(learning_station.coordinates, zone):
                        # Draw book icon at current offset
                        canvas.blit(resource_manager.book_icon, (icon_x, icon_y))
                        # Move right for the next icon, add some spacing
                        icon_x += icon_w + 5
                                
                resource_manager.child_icon_w, resource_manager.child_icon_h = resource_manager.child_icon.get_width(), resource_manager.child_icon.get_height()
                icon_x = rect_x
                icon_y = rect_y

                for child in self.map.children:
                    if self.map.coordinates_in_zone(child.coordinates, zone):
                        # Draw child icon at current offset
                        canvas.blit(resource_manager.child_icon, (icon_x, icon_y))
                        # Move right for the next icon, add some spacing
                        icon_x += resource_manager.child_icon_w + 5  

                min_x, min_y = np.min(zone_coords, axis=0)
                max_x, max_y = np.max(zone_coords, axis=0)

                zone_center = np.array([
                    (min_x + max_x + 1) / 2 * pix_width_size,
                    (min_y + max_y + 1) / 2 * pix_height_size,
                ])

            mid_x = self.map.width // 2 * pix_width_size
            pygame.draw.line(
                canvas,
                (0, 0, 0),  
                (mid_x, 0),
                (mid_x, self.navigation_area_height),
                width=4,
            )

            # --- Draw the agent in the center of its zone ---
            agent_zone = self.map.get_zone(self.agent_coordinates)
            zone_coords = agent_zone.coordinates  

            min_x, min_y = np.min(zone_coords, axis=0)
            max_x, max_y = np.max(zone_coords, axis=0)

            zone_center = np.array([
                (min_x + max_x + 1) / 2 * pix_width_size,
                (min_y + max_y + 1) / 2 * pix_height_size,
            ])

            pygame.draw.circle(
                canvas,
                (0, 0, 0),  # agent in white
                zone_center,
                min(pix_width_size, pix_height_size) / 2.5,
            )

            # --- displaying normatively relevant information area (below grid) ---
            font = pygame.font.Font(None, 24)
            facts = self.get_facts()  
            zone = self.map.get_zone(self.agent_coordinates)
            pygame.draw.rect(self.window, (30, 30, 30), (0, self.navigation_area_height, self.navigation_area_width, facts_area_height))

            y_offset = self.navigation_area_height+zone_name_area_height + 10

            for zone_happening in facts["zones"]:
                text_surface = font.render(zone_happening[1]+" in zone " +zone_happening[2]+".", True, (255, 255, 255))
                self.window.blit(text_surface, (10, y_offset))  
                y_offset += text_surface.get_height() + 5
            
            y_offset += text_surface.get_height() + 5

            for child_condition in facts["children"]:
                text_surface = font.render(child_condition[1]+" in zone " +child_condition[2]+".", True, (255, 255, 255))
                self.window.blit(text_surface, (10, y_offset))  
                y_offset += text_surface.get_height() + 5

            self.window.blit(canvas, (0, zone_name_area_height))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    pygame.quit()
                    sys.exit()

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
