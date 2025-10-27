from GUARDIANCE.contained_DMMs.DMM_components.RL_agents.navigation_agent import setup_agent  
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.MAT_mapping_PG import MAT_mapping_PG
from preschool.config import Config
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.data_processor_PG import Data_Processor_PG
import uuid
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.guard_PT import Guard_PT
from GUARDIANCE.contained_DMMs.DMM_components.baml.baml_client import b
import torch
import numpy as np

rl_agent_name = "navigation_agent_700_episodes"
    
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
        self.guard = Guard_PT(self.DMM, self.mat_mapping)
        self.static_env_info = None
        self.target_coordinate = None
        self.guiding_rules = None
        self.normative_reasons = {}

    def set_target_rl(self, rl_obs):
        #width = self.static_env_info["size"][0]
        #height = self.static_env_info["size"][1]
        # # target position (one-hot encoding)
        #target_window = np.zeros((height, width))
        #target_x, target_y = self.target_coordinate
        #target_window[target_y, target_x] = 1
        #target_flat = target_window.flatten()
#
        #nn_input = np.concatenate([rl_obs, target_flat])
#
        ##rl_obs["target_window"] = target_window
        #return nn_input
        return np.array([rl_obs[0], rl_obs[1], self.target_coordinate[0], self.target_coordinate[1]])

    def navigate(self, rl_obs):
        assert self.target_coordinate is not None, "Target position not set."
        rl_obs = self.set_target_rl(rl_obs)
        with torch.no_grad():
            move_dir = self.DMM.rl_agent.policy_dqn(self.DMM.rl_agent.transformation(rl_obs)).argmax().item()
        return ("move", move_dir)

    def DMM_take_action(self, rl_obs, extracted_data, new_plan):

        DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)

        agent_coordinate = np.array([DMM_observation["agent_coordinate"]["x"], DMM_observation["agent_coordinate"]["y"]])
        if np.array_equal(agent_coordinate, self.target_coordinate):
            self.target_coordinate = None
        #follow course of action until target reached or normative reasons changed
        if self.target_coordinate is not None and not new_plan:
            action = self.navigate(rl_obs)
        else:
            output =self.DMM.LLM.Take_Action_PG(agent_coordinate=DMM_observation["agent_coordinate"], station_coordinates=DMM_observation["station_coordinates"], zones=DMM_observation["zones"], child_conditions=DMM_observation["child_conditions"], happenings=DMM_observation["happenings"])
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
        
    def take_action(self, rl_obs, observation):
        extracted_data = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation, self.static_env_info)
        if self.guiding_rules is None:
            self.guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
        test = extracted_data["children"] | extracted_data["happenings"] 
        if not extracted_data["children"] | extracted_data["happenings"] == self.normative_reasons:
            self.normative_reasons = extracted_data["children"] | extracted_data["happenings"]
            new_guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
            if new_guiding_rules != self.guiding_rules:
                self.guiding_rules = new_guiding_rules
                action = self.DMM_take_action(rl_obs , extracted_data, True)
            else:
                action = self.DMM_take_action(rl_obs , extracted_data, False)

        else:
            action = self.DMM_take_action(rl_obs , extracted_data, False)
            #guard_observation = self.data_processor.guard_observation(extracted_data, self.guiding_rules)
            #action = self.guard.ensure_conformity(action, self.guiding_rules, guard_observation)
        return action
                
        #self.normative_reasons = extracted_data["children"] | extracted_data["happenings"]
        #if extracted_data["children"] or extracted_data["happenings"]:
#
        ## pass information about the environment to the reasoning unit to get the moral obligations
        #self.guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
        #DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)
        #action = self.DMM_take_action(rl_obs , DMM_observation)
        ##guard_observation = self.data_processor.guard_observation(extracted_data, self.guiding_rules)
        ##action = self.guard.ensure_conformity(action, self.guiding_rules, guard_observation)
        #return action
