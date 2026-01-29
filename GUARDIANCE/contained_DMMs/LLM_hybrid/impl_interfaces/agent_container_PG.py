from GUARDIANCE.interfaces.agent_container import Agent_Container
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
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.LLM_hybrid import LLM_hybrid


action_to_direction = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}

logger = logging.getLogger(__name__)


class Agent_Container_PG(Agent_Container):
    def __init__(self):
        self.mat_mapping = MAT_mapping_PG()
        self.data_processor = Data_Processor_PG()
        self.DMM = LLM_hybrid()
        self.reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.guard = Guard_PG(self.mat_mapping)
        

        self.static_env_info = None

        # normatively relevant information used for applying a reason theory to ensure normative conformity
        self.normative_reasons = None
        self.guiding_rules = set()


    def take_action(self, observation):
        rl_obs, observation = observation
        
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
                DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)
                DMM_input = {"reasons_changed": True,
                            "rl_obs": rl_obs,
                            "DMM_observation": DMM_observation,
                            "guiding_rules": new_guiding_rules
                                }
                # let the DMM determine a new course of action that is conform with the change in guiding rules
                action = self.DMM.take_action(DMM_input)
            else:
                DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)
                DMM_input = {"reasons_changed": False,
                            "rl_obs": rl_obs,
                            "DMM_observation": DMM_observation,
                            "guiding_rules": self.guiding_rules
                            }
                action = self.DMM.take_action(DMM_input)
        else:
            DMM_observation = self.data_processor.DMM_observation(extracted_data, self.guiding_rules)
            DMM_input = {"reasons_changed": False,
                        "rl_obs": rl_obs,
                        "DMM_observation": DMM_observation,
                        "guiding_rules": self.guiding_rules
                        }
            action = self.DMM.take_action(DMM_input)

        guard_observation = self.data_processor.guard_observation(extracted_data, self.guiding_rules)
        violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
        if violated_obligation:
            self.DMM.add_feedback(violated_obligation, action)
            action = self.DMM.retrigger(rl_obs)
            violated_obligation = self.guard.violated_obligation(action, self.guiding_rules, guard_observation)
            if violated_obligation:
                MATs = [(rule[0][1],rule[1]) for rule in self.guiding_rules]
                action = self.mat_mapping.default_action(MATs, guard_observation)
            self.course_correction = True

        return action
