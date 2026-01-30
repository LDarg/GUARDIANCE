from GUARDIANCE.interfaces.DMM import DMM
import logging
import uuid

# baml interface for API calls to LLMs 
from GUARDIANCE.contained_DMMs.DMM_components.baml.baml_client import b

logger = logging.getLogger(__name__)

class LLM(DMM):
    def __init__(self):
        self.LLM = b

    def take_action(self, DMM_input):
        DMM_observation = DMM_input
        output =self.LLM.Take_Action_Preschool(agent_zone=DMM_observation["agent_zone"], station_zones=DMM_observation["stations_zones"], zone_ids= DMM_observation["zone_ids"], child_conditions=DMM_observation["child_conditions"], happenings=DMM_observation["happenings"])
        action = self.output_to_action(output)
        return action
    
    #Transforms the output to the format expected as input from the environment 
    def output_to_action(self, LLM_Output):
        if LLM_Output.type == "move":
            return (LLM_Output.type, uuid.UUID(LLM_Output.identifier))
        elif LLM_Output.type == "prepare":
            return (LLM_Output.type, None)
        elif LLM_Output.type == "help":
            return (LLM_Output.type, uuid.UUID(LLM_Output.identifier), LLM_Output.help)