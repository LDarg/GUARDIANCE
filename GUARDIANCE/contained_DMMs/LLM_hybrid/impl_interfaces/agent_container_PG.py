from GUARDIANCE.interfaces.agent_container import Agent_Container
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.MAT_mapping_PG import MAT_mapping_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.data_processor_PG import Data_Processor_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.guard_PG import Guard_PG
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.moral_module_PG import Moral_Module_PG
import logging
from GUARDIANCE.contained_DMMs.LLM_hybrid.impl_interfaces.LLM_hybrid import LLM_hybrid

logger = logging.getLogger(__name__)


class Agent_Container_PG(Agent_Container):
    def __init__(self):

        self.mat_mapping = MAT_mapping_PG()
        self.data_processor = Data_Processor_PG()
        self.DMM = LLM_hybrid()
        self.guard = Guard_PG(self.mat_mapping)
        reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.moral_module = Moral_Module_PG(reasoning_unit, self.data_processor)

        # Normatively relevant information used for applying a reason theory to ensure normative conformity
        self.normative_reasons = None
        self.guiding_rules = set()

    def update_static_env_info(self, static_env_info):
        self.moral_module.static_env_info = static_env_info

    def take_action(self, observation):
        rl_obs, observation = observation

        self.moral_module.update_relevant_information(observation)
        self.moral_module.update_guiding_rules()

        DMM_observation = self.moral_module.DMM_observation()

        DMM_input = {
            "rules_changed": self.moral_module.rules_changed,
            "rl_obs": rl_obs,
            "DMM_observation": DMM_observation,
            "guiding_rules": self.moral_module.guiding_rules_cache,
        }

        action = self.DMM.take_action(DMM_input)

        # Call the guard to check whether the action the agent wants to execute is conform with the binding rules; 
        # In case of a violation of noramtive requirements: give feedback initiate course correction
        guard_observation = self.moral_module.guard_observation()
        violated_obligation, action = self.guard.ensure_conformity(action, self.moral_module.guiding_rules_cache, guard_observation)
        if violated_obligation:
            self.DMM.add_feedback(violated_obligation, action)
            action = self.DMM.retrigger(rl_obs)
            self.course_correction = True

        return action
