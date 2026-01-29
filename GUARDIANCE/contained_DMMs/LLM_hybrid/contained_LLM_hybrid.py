from GUARDIANCE.contained_DMMs.DMM_components.RL_agents.navigation_agent import setup_agent  
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.MAT_mapping_PG import MAT_mapping_PG
from preschool.config import Config
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.data_processor_PG import Data_Processor_PG
import uuid
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.guard_PG import Guard_PG
from GUARDIANCE.contained_DMMs.DMM_components.baml.baml_client import b
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

rl_agent_name = "navigation_agent_700_episodes"

action_to_direction = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}
    
class LLM_hybrid():
    def __init__(self, rl_agent_name):
        self.rl_agent,_= setup_agent(rl_agent_name)
        self.LLM = b

"""
An implementation of an RGA (reason-guided agent) with an LLM that can call an RL agent as tool as DMM (decision-making module).
"""
class contained_LLM_PG():
    def __init__(self):
        self.config = Config()
        self.DMM = LLM_hybrid(rl_agent_name)
        self.mat_mapping = MAT_mapping_PG(self.config)
        self.data_processor = Data_Processor_PG()
        self.reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.guard = Guard_PG(self.mat_mapping)
        self.static_env_info = None
        self.target_coordinate = None
        self.guiding_rules = None
        self.normative_reasons = {}
        self.course_correction = False

        self.DMM_input = None #memory

    def set_target_rl(self, rl_obs):
        return np.array([rl_obs[0], rl_obs[1], self.target_coordinate[0], self.target_coordinate[1]])

    def navigate(self, rl_obs):
        assert self.target_coordinate is not None, "Target position not set."
        rl_obs = self.set_target_rl(rl_obs)
        with torch.no_grad():
            move_dir = self.DMM.rl_agent.policy_dqn(self.DMM.rl_agent.transformation(rl_obs)).argmax().item()
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

    def DMM_take_action(self, rl_obs, extracted_data, new_plan):

        DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)
        self.update_LLM_input(DMM_observation)

        # determine if the agent has reached its target position
        #TODO: ab hier ist es vom DMM abhängig 
        agent_coordinate = np.array([DMM_observation["agent_coordinate"]["x"], DMM_observation["agent_coordinate"]["y"]])
        if np.array_equal(agent_coordinate, self.target_coordinate):
            self.target_coordinate = None
        #follow course of action until target reached or normative reasons changed
        if self.target_coordinate is not None and not new_plan:
            action = self.navigate(rl_obs)
        #if the target is reached or the reasons have changed, ask the LLM for a new action plan
        else:
            output =self.DMM.LLM.Take_Action_PG(**self.DMM_input)
            self.chosen_action = output
            action= self.output_to_action(output, rl_obs)

        #ensure conformity through invoking the guard

        return action

    #transforms the output to the format expected as input from the environment 
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
        
    #IMPORTANT: in der grid-world wichtiger, dass keine unnötigen anfragen in jedem schritt an LLM gestellt, weil die trajectories um ein ziel zu erreicehn viel länger sind (viele states, in denen unntige anfragen gestellt werden würden)
    # TODO: das ist nicht abhängig vom konkreten DMM?
    def take_action(self, rl_obs, observation):
        extracted_data = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation, self.static_env_info)
        if self.guiding_rules is None:
            self.guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
        #check if normative reasons have changed
        if not extracted_data["children"] | extracted_data["happenings"] == self.normative_reasons:
            self.normative_reasons = extracted_data["children"] | extracted_data["happenings"]
            new_guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
            # check if the change in normative reasons enforced an adaptation of the guiding rules
            if new_guiding_rules != self.guiding_rules:
                self.guiding_rules = new_guiding_rules
                # let the DMM determine a new course of action that is conform with the change in guiding rules
                action = self.DMM_take_action(rl_obs , extracted_data, True)
            else:
                action = self.DMM_take_action(rl_obs , extracted_data, False)
        else:
            action = self.DMM_take_action(rl_obs , extracted_data, False)

        guard_observation = self.data_processor.guard_observation(extracted_data, self.guiding_rules)
        violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
        if violated_obligation:
            self.add_feedback(violated_obligation, action)
            action = self.retrigger(rl_obs)
            violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
            if violated_obligation:
                MATs = [(rule[0][1],rule[1]) for rule in self.guiding_rules]
                action = self.mat_mapping.default_action(MATs, guard_observation)
            self.course_correction = True

        return action
