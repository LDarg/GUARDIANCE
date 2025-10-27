import numpy as np
import pygame
import sys
from collections import OrderedDict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import yaml
import os
from preschool.resource_manager import ResourceManager
from preschool.config import Config
import uuid

pygame.init()
resource_manager = ResourceManager()  
resource_manager.load_icons(size=(50,50))

class Child():
    def __init__(self, condition, coordinates):
        self.condition = condition
        self.coordinates = coordinates
        self.id = uuid.uuid4()

class Zone():
    def __init__(self, coordinates, name):
        self.name = name
        self.coordinates = coordinates
        self.happening = None
        self.id = uuid.uuid4()

class Learning_Station():
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.steps_until_finished = 1  # takes 1 time-step to finish the learning station
    
    def progress(self):
        self.steps_until_finished -= 1

    def finished(self):
        if self.steps_until_finished <= 0:
            return True

class Map():
    def __init__(self, random_gen, config:Config):
        self.directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.width = 8
        self.height = 8
        self.config = config

        self.happenings = config.happenings

        self.random_gen = random_gen

        self.zones = set()
        self.zones.add(Zone(np.array([(x, y) for x in range(self.width // 2) for y in range(self.height)]), name="A"))
        self.zones.add(Zone(np.array([(x, y) for x in range(self.width // 2, self.width) for y in range(self.height)]), name="B"))
        assert self.width == self.height, "The map must have a quadratic layout."   

        self.children = set()
        self.max_conditions = 3

        self.learning_stations = set()
        self.spawn_learning_stations()

    def random_location(self):
        return np.array([
            self.random_gen.integers(0, self.width),
            self.random_gen.integers(0, self.height),
        ])

    def spawn_learning_stations(self):
        for _ in range(3):  # spawn 3 learning stations
            learning_station = Learning_Station(self.random_location())
            self.learning_stations.add(learning_station)

    def generate_moral_goal(self):
        if len(self.children) <= self.max_conditions:
            child = Child(random.choice(self.config.conditions), (self.random_location()))
            self.children.add(child)
    
    def generate_moral_goal_unique(self):
        if len(self.children) <= self.max_conditions:
            condition = random.choice(self.config.conditions)
            while condition in [child.condition for child in self.children]:
                condition = random.choice(self.config.conditions)
            child = Child(condition, (self.random_location()))
            self.children.add(child)

    def generate_happening(self):
        zone = random.choice(list(self.zones))
        # ensures that only one happening occurs at a time and thus, that there is at least one zone where the agent can move without violating a constraint
        if not any(zone.happening for zone in self.zones):
            zone.happening = random.choice(self.happenings)

    def delete_moral_goal(self, child):
        self.children.remove(child)

    def coordinates_in_zone(self, coordinates, zone):
        for zone_coordinate in zone.coordinates:
            if np.all(zone_coordinate == coordinates):
                return True
        return False

    """
    returns zone for coordinates
    """
    def get_zone(self, coordinates):
        for zone in self.zones:
            for coord in zone.coordinates:
                if np.all(coord == coordinates):
                    return zone
        raise ValueError("Coordinates do not belong to any zone.")

class Preschool_Grid(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 3} 

    def __init__(self,    
                render_mode=None, 
                ):

        self.config = Config()

        self.agent_coordinates = np.array([0,0])
        self.agent_id = uuid.uuid4()
        self.render_mode = render_mode

        self.reset(options=None)
        
        #self.goal_position = (random.randint(self.map.width), random.randint(self.map.height))

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        """
        the observation space:
        the observation is a one hot encoding of the agent position, the person positions and the water tiles flattened to a 1d vector (np.array)
        """
        self.size = self.map.width * self.map.height
        self.grid_width = self.map.width   # number of columns in the grid
        self.grid_height = self.map.height   # number of rows in the grid
        len_obs_dict = len(self.get_obs_dict())
        self.total_grid_cells = self.map.width  * self.map.height
        observation_size = len_obs_dict * self.total_grid_cells
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(observation_size,),
            dtype=np.float32
        )

        """
        the action space (primitive actions the agent can take):
        0:  "right",
        1:  "down",
        2:  "left",
        3:  "up",
        4:  "pull out of water" 
        """
        self.action_space = spaces.Discrete(4)
        self.directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

        """
        mapping of primitive actions from `self.action_space` to
        the direction the agent will walk in if that action is taken.
        """
        self.action_to_direction = {
            0: self.directions[0], #right
            1: self.directions[1], #down
            2: self.directions[2], #left
            3: self.directions[3], #up
        }

        """
        setting parameters for rendering the environment
        """
        cell_width = 100  # width of each cell in pixels
        cell_height = 100  # height of each cell in pixels
        self.navigation_area_width = self.grid_width * cell_width 
        self.navigation_area_height = self.grid_height * cell_height 
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def initilize_map(self, randm_gen):
        self.map = Map(randm_gen)
        self.map.spawn_learning_stations()

    """
    functions for returning information about the state 

    return of the observation function: the flattened and concatenated agent and goal position observations (np.array)
    """
    def observation(self):
        return np.array([self.agent_coordinates[0], self.agent_coordinates[1]])

    def get_obs_dict(self):
        # agent's position (one-hot encoding)
        agent_window = np.zeros((self.map.height, self.map.width))
        agent_x, agent_y = self.agent_coordinates
        agent_window[agent_x, agent_y] = 1

        observation = OrderedDict({
            "agent_window": agent_window
        })
        return observation

    def get_facts(self):
        children_facts = []
        zones_facts = []

        for child in self.map.children:
            children_facts.append((child.id, f"{child.condition}", child.coordinates))

        for zone in self.map.zones:
                if zone.happening is not None:
                    zones_facts.append((zone.id, zone.happening, zone.coordinates))
        
        return {
            "children": children_facts,
            "happenings": zones_facts,
            "agent_coordinate": self.agent_coordinates,
            "station_coordinates": [station.coordinates for station in self.map.learning_stations],
        }
    
    def static_facts(self):
        return {
            "zones": [(zone.id, zone.coordinates) for zone in self.map.zones],
            "size": (self.map.width, self.map.height)
        }
    
    """
    functions for handling positions
    """
    def to_2d_coordinates(self, index, width):
        row = index // width
        col = index % width
        return [col, row]
    
    def to_1d_index(self, position):
        position_1d = position[1] * self.map.width + position[0]
        return position_1d

    
    def random_agent_pos(self, random_gen):
        # choose the agent's location randomly and ensure that the agent doesn't spawn at the goal position
        self.agent_coordinates = np.array([
            random_gen.integers(0, self.map.width),
            random_gen.integers(0, self.map.height),
        ])

    """
    functions for resetting the environment
    """
    def reset(self, seed=None, options=None):
        self.agent_coordinates = np.array([0,0])
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().reset(seed=seed)
        self.initialize_at_reset(self.np_random)
        observation = self.observation()
        info = self.get_facts()
        self.render()

        return observation, info
        
    
    def initialize_at_reset(self, random_gen):
        self.map = Map(random_gen, self.config)
        self.random_agent_pos(random_gen)
            
    """
    update the environment according to what the agent does and subsequently update the environment according to how the persons move
    returns the reward according to the selected reward type (instrumental, rescuing, waiting or MO)
    also returns the output of a cost functions indicating if the agent pushed a person off the bridge
    """
    def step(self, action):
        reward = 0
        terminated = False

        if action[0] == "move":
        # execute the action if it is taking a step in a direction
            direction = self.action_to_direction[action[1]]
            # ensures that the agent's position stays in the grid
            self.agent_coordinates[0] = np.clip(
                self.agent_coordinates[0] + direction[0], 0, self.map.width - 1
            )
            self.agent_coordinates[1] = np.clip(
                self.agent_coordinates[1] + direction[1], 0, self.map.height - 1
            )

        # exceute the action if it is preparing a learning station
        if action[0] == "prepare":
            for learning_station in self.map.learning_stations:
                if np.equal(self.agent_coordinates, learning_station.coordinates).all():
                    learning_station.progress()
                    if learning_station.finished():
                        self.map.learning_stations.remove(learning_station)
                    break

        if action[0] == "help":
            for child in self.map.children:
                if child.id == action[1] and np.equal(self.agent_coordinates, child.coordinates).all():
                    if self.map.config.resolutions[child.condition].replace(" ", "_") == action[2]:
                        self.map.children.remove(child)
                        break

        # exceute the action if it is helping a person or preparing a learning station
        #if action == 4:
        #    to_remove = []
        #    for child in list(self.map.children):  # iterate over a copy
        #        if np.equal(self.agent_coordinates, child.coordinates).all():
        #            to_remove.append(child)
        #    for child in to_remove:
        #        self.map.delete_moral_goal(child)
#
        #    for learning_station in self.map.learning_stations:
        #        if np.equal(self.agent_coordinates, learning_station.coordinates).all():
        #            learning_station.progress()
        #            if learning_station.finished():
        #                self.map.learning_stations.remove(learning_station)
        #            break

        # generate moral goals and happenings with a certain probability
        if random.random() < 0.5: #0.15
            self.map.generate_moral_goal()
        if random.random() < 0: #0.1
            self.map.generate_happening()

        observation = self.observation()
        info = self.get_facts()

        self.render()
    
        return observation, reward, terminated, False, info
    
    
    """
    functions for rendering the envrionment 
    """
    
    def render(self):
        if self.render_mode == "human":
            canvas = self.render_frame()
            self.handle_events()
            self.window.blit(canvas, (0, 0))
            self.update_display()

    def handle_events(self):
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    pygame.quit()
                    sys.exit()

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            

    def render_frame(self):
        facts_area_height = 300
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Preschool")
            self.window = pygame.display.set_mode(
                (self.navigation_area_width, self.navigation_area_height+facts_area_height)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear window
        self.window.fill((0, 0, 0))

        # Create canvas on which the grid is drawn
        canvas = pygame.Surface((self.navigation_area_width, self.navigation_area_height))
        canvas.fill((0, 0, 255))  # fallback background

        pix_width_size = self.navigation_area_width / self.map.width
        pix_height_size = self.navigation_area_height / self.map.height

        # --- zones ---
        for x in range(self.map.width):
            for y in range(self.map.height):
                zone = self.map.get_zone(np.array([x, y]))
                #Set colors for zones
                if zone.name == "A":
                    color = (255, 255, 197)
                elif zone.name == "B":
                    color = (173, 216, 229)
                if zone.happening is not None:
                    color = (255, 0, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (int(x * pix_width_size), int(y * pix_height_size)),
                        (int(pix_width_size), int(pix_height_size)),
                    ),
                )

        # --- children + agent ---
        agent_pos = ((self.agent_coordinates + 0.5) * np.array([pix_width_size, pix_height_size])).astype(int)
        smaller_radius = int(min(pix_width_size, pix_height_size) / 6)
        color_children = (255, 165, 0)
        color_agent= (0, 0, 0)

        for child in self.map.children:
            if np.array_equal(self.agent_coordinates, child.coordinates):
                left_pos = (agent_pos - np.array([smaller_radius, 0])).astype(int)
                right_pos = (agent_pos + np.array([smaller_radius, 0])).astype(int)
                #canvas.blit(resource_manager.child_icon, (left_pos, right_pos))
                pygame.draw.circle(canvas, color_agent, tuple(left_pos), smaller_radius)
                pygame.draw.circle(canvas, color_children, tuple(right_pos), smaller_radius)
            else:
                child_pos = ((child.coordinates + 0.5) * np.array([pix_width_size, pix_height_size])).astype(int)
                #canvas.blit(resource_manager.child_icon, (child_pos[0],child_pos[1]))
                pygame.draw.circle(canvas, color_children, tuple(child_pos), int(min(pix_width_size, pix_height_size) / 3))

        if not any(np.array_equal(self.agent_coordinates, child.coordinates) for child in self.map.children):
            pygame.draw.circle(canvas, color_agent, tuple(agent_pos), int(min(pix_width_size, pix_height_size) / 3))

        # --- learning stations ---
        for learning_station in self.map.learning_stations:
            ls_pos = ((learning_station.coordinates + 0.5) * np.array([pix_width_size, pix_height_size])).astype(int)
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (ls_pos[0] - int(pix_width_size / 6), ls_pos[1] - int(pix_height_size / 6)),
                    (int(pix_width_size / 3), int(pix_height_size / 3)),
                ),
            )

        # --- gridlines ---
        for x in range(self.map.width + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (int(pix_width_size * x), 0),
                (int(pix_width_size * x), self.navigation_area_height),
                width=1,
            )
        for y in range(self.map.height + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, int(pix_height_size * y)),
                (self.navigation_area_width, int(pix_height_size * y)),
                width=1,
            )

        font = pygame.font.Font(None, 24)
        facts = self.get_facts()  # get the list of facts
        zone = self.map.get_zone(self.agent_coordinates)
        pygame.draw.rect(self.window, (30, 30, 30), (0, self.navigation_area_height, self.navigation_area_width, facts_area_height))

        y_offset = self.navigation_area_height + 10

        for fact in facts["children"]:
            text_surface = font.render(fact[1]+": " +str(fact[2]), True, (255, 255, 255))
            self.window.blit(text_surface, (10, y_offset))   # <-- still using self.window here
            y_offset += text_surface.get_height() + 5

        # Blit grid to window
        #self.window.blit(canvas, (0, 0))
        return canvas

