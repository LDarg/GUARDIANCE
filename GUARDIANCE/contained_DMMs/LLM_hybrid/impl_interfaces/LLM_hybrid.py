from GUARDIANCE.interfaces.DMM import DMM
import logging
from GUARDIANCE.contained_DMMs.DMM_components.RL_agents.navigation_agent import setup_agent  
import torch
import numpy as np
import uuid

# baml interface for API calls to LLMs (part of the agent that decides on the course of action)
from GUARDIANCE.contained_DMMs.DMM_components.baml.baml_client import b

# Reinforcement Learning agent (called by the LLM to navigate in the environment)
rl_agent_name = "navigation_agent_700_episodes"
rl_agent,_= setup_agent(rl_agent_name)

action_to_direction = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}

logger = logging.getLogger(__name__)

class LLM_hybrid(DMM):
    def __init__(self):
        self.LLM = b
        self.rl_agent= rl_agent
        self.target_coordinate = None

        self.DMM_input = None

    def take_action(self, DMM_input):
        
        DMM_observation = DMM_input["DMM_observation"]
        guiding_rules = DMM_input["guiding_rules"]
        reasons_changed = DMM_input["reasons_changed"]
        rl_obs = DMM_input["rl_obs"]

        self.update_LLM_input(DMM_observation)

        agent_coordinate = np.array([DMM_observation["agent_coordinate"]["x"], DMM_observation["agent_coordinate"]["y"]])
        
        if np.array_equal(agent_coordinate, self.target_coordinate):
            self.target_coordinate = None
        #Follow course of action until target position is reached or normative reasons changed
        if self.target_coordinate is not None and not reasons_changed:
            action = self.navigate(rl_obs)
        #If the target is reached or the reasons have changed, ask the LLM for a new action plan
        else:
            output =self.LLM.Take_Action_PG(**self.DMM_input)
            self.chosen_action = output
            action= self.output_to_action(output, rl_obs)

        return action
    

    def set_target_rl(self, rl_obs):
        return np.array([rl_obs[0], rl_obs[1], self.target_coordinate[0], self.target_coordinate[1]])

    def navigate(self, rl_obs):
        assert self.target_coordinate is not None, "Target position not set."
        rl_obs = self.set_target_rl(rl_obs)
        with torch.no_grad():
            move_dir = self.rl_agent.policy_dqn(self.rl_agent.transformation(rl_obs)).argmax().item()
        return ("move", move_dir)
    
    def update_LLM_input(self, DMM_observation, feedback=None):
        self.DMM_input = {
            "agent_coordinate": DMM_observation["agent_coordinate"],
            "station_coordinates": DMM_observation["station_coordinates"],
            "zones": DMM_observation["zones"],
            "child_conditions": DMM_observation["child_conditions"],
            "happenings": DMM_observation["happenings"]
        }
        if feedback:
            self.DMM_input["feedback"] = feedback

    #Tranforms the output to the format expected as input from the environment 
    def output_to_action(self, LLM_Output, rl_obs):
        if LLM_Output.type == "move":
            self.target_coordinate = np.array(LLM_Output.target_coordinate)
            action = self.navigate(rl_obs)
            return action
        elif LLM_Output.type == "prepare":
            return (LLM_Output.type, None)
        elif LLM_Output.type == "help":
            return (LLM_Output.type, uuid.UUID(LLM_Output.identifier), LLM_Output.help)
        elif LLM_Output.type == "idle":
            return (LLM_Output.type, None)
        
    def primitive_to_LLM_input(self, action):
        if action[0] == "move":
            input = {"type": "move", "direction": action_to_direction[action[1]]}
        if action[0] == "prepare":
            input = {"type": "prepare"}
        if action[0] == "help":
            input = {"type": "help", "child_id": str(action[1]), "help": action[2]}
        if action[0] == "idle":
            input = {"type": "idle"}
        return input
    
    def retrigger(self, rl_obs):
        output = b.Take_Action_PG(**self.DMM_input)
        action = self.output_to_action(output, rl_obs)
        return action

    def add_feedback(self, violated_obligation, action):
        relevant_state_elements = {"agent_coordinate": self.DMM_input["agent_coordinate"],
                 "child_conditions": self.DMM_input["child_conditions"],
                 "happenings": self.DMM_input["happenings"]}
        action = self.primitive_to_LLM_input(action)
        violated_obligation = {"id": str(violated_obligation[1]), "required_MAT": violated_obligation[0]}
        feedback = {"state": relevant_state_elements, "action": action, "violated_obligation": violated_obligation}
        if "feedback" in self.DMM_input:
              self.DMM_input["feedback"].append(feedback)
        else:
            self.DMM_input["feedback"] = [feedback]
        logger.info(f"feedback added: {feedback}")
        pass
