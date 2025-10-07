from LLM_agent.DMM.baml.baml_client import b
from GUARDIANCE.reasoning_unit import ReasoningUnit
from LLM_agent.impl_interfaces.MAT_mapping_PT import MAT_mapping_PT
from preschool.config import Config
from LLM_agent.impl_interfaces.data_processor_PT import Data_Processor_PT
import uuid
from LLM_agent.impl_interfaces.guard_PT import Guard

"""
An implementation of an RGA (reason-guided agent) with an LLM as DMM (decision-making module).
"""
class RGA():
    def __init__(self):
        self.config = Config()
        self.LLM_interface = b
        self.mat_mapping = MAT_mapping_PT(self.config)
        self.data_processor = Data_Processor_PT()
        self.reasoning_unit = ReasoningUnit(self.mat_mapping, self.data_processor)
        self.guard = Guard(self.LLM_interface, self.mat_mapping)

    #transforms the output to the format expected as input from the environment 
    def output_to_action(self, LLM_Output):
        if LLM_Output.type == "move":
            return (LLM_Output.type, uuid.UUID(LLM_Output.identifier))
        elif LLM_Output.type == "prepare":
            return (LLM_Output.type, None)
        elif LLM_Output.type == "help":
            return (LLM_Output.type, uuid.UUID(LLM_Output.identifier), LLM_Output.help)
        
    def take_action(self, observation):
        extracted_data = self.data_processor.extract_relevant_information(self.reasoning_unit.reason_theory, observation)
        # pass information about the environment to the reasoning unit to get the moral obligations
        guiding_rules = self.reasoning_unit.moral_obligations(extracted_data)
        DMM_observation = self.data_processor.DMM_observation(extracted_data, guiding_rules)
        station_zones = DMM_observation["stations_zones"]
        output =b.Take_Action_Preschool(agent_zone=DMM_observation["agent_zone"], station_zones=DMM_observation["stations_zones"], zone_ids= DMM_observation["zone_ids"], child_conditions=DMM_observation["child_conditions"], happenings=DMM_observation["happenings"])
        guard_observation = self.data_processor.guard_observation(extracted_data, guiding_rules)
        action = self.output_to_action(output)
        self.guard.ensure_conformity(action, guiding_rules, guard_observation)
        return action