from preschool.config import Config
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
import uuid
import numpy as np

# Configuration for the preschool environment including inforamtion about which MAT is required for helping children 
config = Config()

"""
A MAT-mapping for the text-version of the preschool setting navigated by an LLM agent.
"""
class MAT_mapping_PG(MAT_Mapping):
    def __init__(self):
        self.config = config

    """
    Checks if the agent exits the zone if currently inside and does not enter the zone if currently outside.
    """
    def obligation_violated(self, action, MAT, observation):

        if MAT[0] == "Stay_out_of_the_zone":
            y_coordinate_boundary = 3
            zone = next(zone for zone in observation["zones"] if zone["zone_id"] == MAT[1])
            if any(np.array_equal(coord, np.array([0, 0])) for coord in zone["coordinates"]):
                #Check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 0):
                    return MAT
                #Check if the agent is about to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary+1 and action == ("move", 2):
                    return MAT
            else:
                # Check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 2):
                    return MAT
                # Check if the agent is about to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary and action == ("move", 0):
                    return MAT
                
    """
    Idling ensures that the agent does not run against a child in a forbidden zone; the guard does not implement alternative strategies for normative goals.
    """
    def default_action(self, MATs, observation):
        return ('idle',)
    
    """
    To determine conflicts (in Horty’s formalization of reasoning), MATs are checked for compliance.
    """
    def execution_conflicted(self, set_of_rules, extracted_data):
        forbidden_zones = []
        for rule in set_of_rules:
            if rule[0][1] == "Stay_out_of_the_zone":
                forbidden_zones += [zone for zone in extracted_data["zones"].values() if zone["zone_id"] == rule[1]]
        # The agent must not be in both zones; otherwise the oblgiations are conflcited
        if len(forbidden_zones) >= 2:
            return True

        return False
