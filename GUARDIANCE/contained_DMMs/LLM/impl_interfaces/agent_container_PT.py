from GUARDIANCE.interfaces.agent_container import Agent_Container
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.MAT_mapping_PT import MAT_mapping_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.data_processor_PT import Data_Processor_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.guard_PT import Guard_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.moral_module_PT import Moral_Module_PT
import logging
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.LLM import LLM

logger = logging.getLogger(__name__)

class Agent_Container_PT(Agent_Container):
    def __init__(self):
        self.mat_mapping = MAT_mapping_PT()
        self.data_processor = Data_Processor_PT()
        self.DMM = LLM()
        reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.moral_module = Moral_Module_PT(reasoning_unit, self.data_processor)
        self.guard = Guard_PT(self.mat_mapping)


    def take_action(self, observation):
        # Update normatively relevant information and guiding rules based on the new observation
        self.moral_module.update_relevant_information(observation)
        self.moral_module.update_guiding_rules()

        # Let the DMM choose an action 
        DMM_observation = self.moral_module.DMM_observation()
        action = self.DMM.take_action(DMM_observation)

        # Invoke the guard to ensure that the action chosen by the DMM is conform with the guiding rules 
        guard_observation = self.moral_module.guard_observation()
        action = self.guard.ensure_conformity(action, self.moral_module.guiding_rules_cache, guard_observation)

        return action

