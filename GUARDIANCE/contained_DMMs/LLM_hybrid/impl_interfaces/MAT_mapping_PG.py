from preschool.config import Config
from GUARDIANCE.interfaces.MAT_mapping import MAT_Mapping
import uuid
import numpy as np

config = Config()

"""
An MAT_mapping for the text-version of the preschool setting navigated by an LLM agent.
"""
class MAT_mapping_PG(MAT_Mapping):
    def __init__(self, config):
        # config is needed for retrieving the infomration which action is required for which child condition 
        self.config = config

    # self.action_to_direction = {
    #        0: self.directions[0], #right
    #        1: self.directions[1], #down
    #        2: self.directions[2], #left
    #        3: self.directions[3], #up
    #    }
    
    def manhattan_distance(array1, array2):
        return np.sum(np.abs(array1 - array2))

    """
    checks if the agent exits the zone if currently inside and does not enter the zone if currently outside
    all trajectories are compliant with each normative goal as none of them has a time constraint
    """
    def obligation_violated(self, action, MAT, observation):

        if MAT[0] == "Stay_out_of_the_zone":
            y_coordinate_boundary = 3
            zone = next(zone for zone in observation["zones"] if zone["zone_id"] == MAT[1])
            # check if the agent is in the left zone
            if any(np.array_equal(coord, np.array([0, 0])) for coord in zone["coordinates"]):
                #check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 0):
                    return MAT
                #check if the agent wants to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary+1 and action == ("move", 2):
                    return MAT
            # the agent is in the right zone
            else:
                # check if the agent does not move out of it
                if any(np.array_equal(observation["agent_coordinate"], coord) for coord in zone["coordinates"]) and action != ("move", 2):
                    return MAT
                # check if the agent wants to enter the zone
                if observation["agent_coordinate"][0] == y_coordinate_boundary and action == ("move", 0):
                    return MAT
                
    """
    Idling ensures that the agent does not run against a child in a forbidden zone; no alternative strategies for normative goals provided.
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
        # both zones must not be entered; but the agent has to stay in one 
        if len(forbidden_zones) >= 2:
            return True

        return False
