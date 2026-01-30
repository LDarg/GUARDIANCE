from GUARDIANCE.interfaces.agent_container import Agent_Container
from GUARDIANCE.reasoning_unit import ReasoningUnit
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.MAT_mapping_PT import MAT_mapping_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.data_processor_PT import Data_Processor_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.guard_PT import Guard_PT
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.data_processor_PT import Data_Processor_PT
import logging
from GUARDIANCE.contained_DMMs.LLM.impl_interfaces.LLM import LLM

logger = logging.getLogger(__name__)

class Agent_Container_PT(Agent_Container):
    def __init__(self):
        self.mat_mapping = MAT_mapping_PT()
        self.data_processor = Data_Processor_PT()
        self.DMM = LLM()
        self.reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.guard = Guard_PT(self.mat_mapping)


    def take_action(self, observation):
        extracted_data = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation)
        guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)

        DMM_observation = self.data_processor.DMM_observation(extracted_data, guiding_rules)
        # Let the DMM choose an action 
        action = self.DMM.take_action(DMM_observation)

        # Invoke the guard to ensure that the action chosen by the DMM is conform with the guiding rules 
        guard_observation = self.data_processor.guard_observation(extracted_data, guiding_rules)
        action = self.guard.ensure_conformity(action, guiding_rules, guard_observation)
        return action

